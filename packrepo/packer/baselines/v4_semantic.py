"""V4: Semantic Baseline Implementation."""

from __future__ import annotations

import logging
import numpy as np
from typing import List, Dict, Any, Optional

from .base import BaselineSelector, BaselineConfig
from ..chunker.base import Chunk
from ..selector.base import SelectionResult

logger = logging.getLogger(__name__)


class V4SemanticBaseline(BaselineSelector):
    """
    V4: Semantic Baseline - Use embedding-based similarity ranking.
    
    Implements semantic similarity using sentence transformers or similar
    embedding models for query-document semantic matching. This provides
    a strong baseline that understands semantic relationships between code
    and represents modern neural information retrieval approaches.
    
    This should be the most challenging baseline to beat, as it leverages
    deep semantic understanding of code relationships.
    """
    
    def __init__(self, tokenizer=None, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic baseline with embedding model.
        
        Args:
            tokenizer: Optional tokenizer for token counting
            model_name: Name of the sentence transformer model to use
        """
        super().__init__(tokenizer)
        self.model_name = model_name
        self._embedding_model = None
        self._chunk_embeddings: Dict[str, np.ndarray] = {}
        self._query_embedding: Optional[np.ndarray] = None
    
    def get_variant_id(self) -> str:
        """Get the baseline variant identifier."""
        return "V4"
    
    def get_description(self) -> str:
        """Get human-readable description of baseline."""
        return f"Semantic similarity baseline (model: {self.model_name})"
    
    def _load_embedding_model(self):
        """Load the sentence transformer model lazily."""
        if self._embedding_model is not None:
            return
        
        try:
            # Try to import sentence-transformers
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"V4: Loading embedding model: {self.model_name}")
            self._embedding_model = SentenceTransformer(self.model_name)
            logger.info("V4: Embedding model loaded successfully")
            
        except ImportError:
            logger.warning(
                "V4: sentence-transformers not available, falling back to simple similarity"
            )
            self._embedding_model = "fallback"
        except Exception as e:
            logger.warning(f"V4: Failed to load embedding model: {e}, using fallback")
            self._embedding_model = "fallback"
    
    def _extract_semantic_content(self, chunk: Chunk) -> str:
        """
        Extract meaningful semantic content from a chunk for embedding.
        
        Args:
            chunk: Chunk to extract content from
            
        Returns:
            Cleaned text suitable for semantic embedding
        """
        content_parts = []
        
        # Add chunk name and file path for context
        content_parts.append(f"File: {chunk.rel_path}")
        content_parts.append(f"Name: {chunk.name}")
        
        # Add docstring if available (most semantic information)
        if chunk.docstring and len(chunk.docstring.strip()) > 0:
            content_parts.append(f"Documentation: {chunk.docstring.strip()}")
        
        # Add signature if available (API information)
        if chunk.signature and len(chunk.signature.strip()) > 0:
            content_parts.append(f"Signature: {chunk.signature.strip()}")
        
        # Add content, but limit length to avoid overwhelming the embedding
        if chunk.content:
            # Take first 500 characters of content for semantic analysis
            content_preview = chunk.content[:500]
            
            # Try to break at a reasonable point (end of line)
            if len(chunk.content) > 500:
                newline_pos = content_preview.rfind('\n')
                if newline_pos > 200:  # Ensure we have reasonable content
                    content_preview = content_preview[:newline_pos]
                content_preview += "..."
            
            content_parts.append(f"Content: {content_preview}")
        
        # Add language for context
        if chunk.language:
            content_parts.append(f"Language: {chunk.language}")
        
        return " | ".join(content_parts)
    
    def _compute_embeddings(self, chunks: List[Chunk]) -> None:
        """
        Compute embeddings for all chunks.
        
        Args:
            chunks: Chunks to compute embeddings for
        """
        logger.info(f"V4: Computing embeddings for {len(chunks)} chunks")
        
        self._load_embedding_model()
        
        if self._embedding_model == "fallback":
            # Use simple hash-based similarity as fallback
            self._compute_fallback_embeddings(chunks)
            return
        
        # Extract semantic content for each chunk
        chunk_texts = []
        chunk_ids = []
        
        for chunk in chunks:
            semantic_text = self._extract_semantic_content(chunk)
            chunk_texts.append(semantic_text)
            chunk_ids.append(chunk.id)
        
        try:
            # Compute embeddings in batches for efficiency
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[i:i + batch_size]
                batch_embeddings = self._embedding_model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    normalize_embeddings=True  # Normalize for cosine similarity
                )
                all_embeddings.extend(batch_embeddings)
            
            # Store embeddings
            for chunk_id, embedding in zip(chunk_ids, all_embeddings):
                self._chunk_embeddings[chunk_id] = embedding
                
            logger.info(f"V4: Computed {len(all_embeddings)} embeddings")
            
        except Exception as e:
            logger.warning(f"V4: Failed to compute embeddings: {e}, using fallback")
            self._compute_fallback_embeddings(chunks)
    
    def _compute_fallback_embeddings(self, chunks: List[Chunk]) -> None:
        """
        Compute simple hash-based embeddings as fallback.
        
        Args:
            chunks: Chunks to compute fallback embeddings for
        """
        logger.info("V4: Using fallback hash-based embeddings")
        
        for chunk in chunks:
            # Create a simple feature vector based on chunk properties
            features = []
            
            # File path features (simple hash of path components)
            path_parts = chunk.rel_path.split('/')
            for part in path_parts:
                features.append(hash(part.lower()) % 1000)
            
            # Pad or truncate to fixed size
            features = (features + [0] * 10)[:10]
            
            # Add other features
            features.extend([
                hash(chunk.name.lower()) % 1000,
                hash(chunk.language.lower()) % 1000 if chunk.language else 0,
                int(chunk.doc_density * 1000) if chunk.doc_density else 0,
                int(chunk.complexity_score) if chunk.complexity_score else 0,
                chunk.line_count % 100,
            ])
            
            # Convert to normalized numpy array
            embedding = np.array(features, dtype=np.float32)
            embedding = embedding / np.linalg.norm(embedding) if np.linalg.norm(embedding) > 0 else embedding
            
            self._chunk_embeddings[chunk.id] = embedding
    
    def _generate_query_embedding(self, chunks: List[Chunk]) -> np.ndarray:
        """
        Generate a query embedding representing the overall repository context.
        
        Args:
            chunks: All available chunks for context
            
        Returns:
            Query embedding vector
        """
        if self._embedding_model == "fallback":
            # Simple centroid of all chunk embeddings
            if self._chunk_embeddings:
                embeddings = list(self._chunk_embeddings.values())
                centroid = np.mean(embeddings, axis=0)
                return centroid / np.linalg.norm(centroid) if np.linalg.norm(centroid) > 0 else centroid
            else:
                return np.zeros(15, dtype=np.float32)  # Match fallback embedding size
        
        # Generate repository summary for semantic query
        # Collect important information about the repository
        file_types = set()
        languages = set()
        important_names = []
        docstrings = []
        
        for chunk in chunks:
            # Collect file extensions
            if '.' in chunk.rel_path:
                ext = chunk.rel_path.split('.')[-1].lower()
                file_types.add(ext)
            
            # Collect languages
            if chunk.language:
                languages.add(chunk.language.lower())
            
            # Collect important function/class names
            if chunk.name and len(chunk.name) > 3:  # Avoid very short names
                important_names.append(chunk.name)
            
            # Collect docstrings for semantic understanding
            if chunk.docstring and len(chunk.docstring.strip()) > 20:
                docstrings.append(chunk.docstring.strip())
        
        # Create query text from repository characteristics
        query_parts = []
        
        if languages:
            query_parts.append(f"Programming languages: {', '.join(sorted(languages))}")
        
        if file_types:
            query_parts.append(f"File types: {', '.join(sorted(file_types))}")
        
        # Add most common important names
        name_counts = {}
        for name in important_names:
            name_counts[name.lower()] = name_counts.get(name.lower(), 0) + 1
        
        common_names = sorted(name_counts.items(), key=lambda x: -x[1])[:10]
        if common_names:
            names = [name for name, _ in common_names]
            query_parts.append(f"Key components: {', '.join(names)}")
        
        # Add sample docstrings for semantic context
        if docstrings:
            # Take first few docstrings as examples
            sample_docs = docstrings[:3]
            for i, doc in enumerate(sample_docs):
                # Limit docstring length
                doc_preview = doc[:200] + "..." if len(doc) > 200 else doc
                query_parts.append(f"Example documentation {i+1}: {doc_preview}")
        
        query_text = " | ".join(query_parts)
        
        try:
            query_embedding = self._embedding_model.encode(
                [query_text],
                show_progress_bar=False,
                normalize_embeddings=True
            )[0]
            
            logger.info(f"V4: Generated query embedding from repository context")
            logger.debug(f"V4: Query text preview: {query_text[:200]}...")
            
            return query_embedding
            
        except Exception as e:
            logger.warning(f"V4: Failed to generate query embedding: {e}")
            # Fallback to centroid
            if self._chunk_embeddings:
                embeddings = list(self._chunk_embeddings.values())
                centroid = np.mean(embeddings, axis=0)
                return centroid / np.linalg.norm(centroid) if np.linalg.norm(centroid) > 0 else centroid
            else:
                return np.zeros(384, dtype=np.float32)  # Default size for MiniLM
    
    def _calculate_similarity_scores(self, query_embedding: np.ndarray) -> Dict[str, float]:
        """
        Calculate cosine similarity scores between query and all chunks.
        
        Args:
            query_embedding: Query vector to compare against
            
        Returns:
            Dictionary mapping chunk IDs to similarity scores
        """
        scores = {}
        
        for chunk_id, chunk_embedding in self._chunk_embeddings.items():
            # Cosine similarity (embeddings are normalized)
            similarity = np.dot(query_embedding, chunk_embedding)
            scores[chunk_id] = float(similarity)
        
        return scores
    
    def _select_baseline(self, chunks: List[Chunk], config: BaselineConfig) -> SelectionResult:
        """
        Select chunks using semantic similarity ranking.
        
        Selection strategy:
        1. Compute embeddings for all chunks
        2. Generate repository-wide query embedding
        3. Score all chunks using cosine similarity
        4. Select highest-similarity chunks until budget exhausted
        
        Args:
            chunks: Preprocessed chunks to select from
            config: Baseline configuration
            
        Returns:
            Selection result with semantically ranked chunks
        """
        selected = []
        selected_ids = set()
        chunk_modes = {}
        selection_scores = {}
        total_tokens = 0
        iterations = 0
        
        if not chunks:
            return SelectionResult(
                selected_chunks=[],
                chunk_modes={},
                selection_scores={},
                total_tokens=0,
                budget_utilization=0.0,
                coverage_score=0.0,
                diversity_score=0.0,
                iterations=0,
            )
        
        logger.info(f"V4: Starting semantic selection from {len(chunks)} chunks")
        
        # Compute embeddings for all chunks
        self._compute_embeddings(chunks)
        
        # Generate query embedding from repository context
        query_embedding = self._generate_query_embedding(chunks)
        
        # Calculate similarity scores
        similarity_scores = self._calculate_similarity_scores(query_embedding)
        
        # Sort chunks by similarity score (descending)
        sorted_chunks = sorted(chunks, key=lambda c: similarity_scores.get(c.id, 0.0), reverse=True)
        
        # Log top scores for debugging
        top_scores = [(c.rel_path, similarity_scores.get(c.id, 0.0)) for c in sorted_chunks[:5]]
        logger.info(f"V4: Top similarity scores: {[(path, f'{score:.3f}') for path, score in top_scores]}")
        
        # Select chunks in similarity order until budget exhausted
        for chunk in sorted_chunks:
            if total_tokens >= config.token_budget:
                break
                
            iterations += 1
            score = similarity_scores.get(chunk.id, 0.0)
            
            # Try full mode first
            tokens = self._get_chunk_tokens(chunk, "full")
            
            # Hard budget constraint: zero overflow allowed
            if total_tokens + tokens > config.token_budget:
                # Try signature mode for large files
                signature_tokens = self._get_chunk_tokens(chunk, "signature")
                if total_tokens + signature_tokens <= config.token_budget:
                    selected.append(chunk)
                    selected_ids.add(chunk.id)
                    chunk_modes[chunk.id] = "signature"
                    selection_scores[chunk.id] = score
                    total_tokens += signature_tokens
                    
                    logger.debug(f"V4: Using signature mode for {chunk.rel_path} (sim={score:.3f})")
                else:
                    # Skip if even signature doesn't fit
                    logger.debug(f"V4: Skipping {chunk.rel_path} - too large even in signature mode")
                continue
            
            # Add chunk in full mode
            selected.append(chunk)
            selected_ids.add(chunk.id)
            chunk_modes[chunk.id] = "full"
            selection_scores[chunk.id] = score
            total_tokens += tokens
            
            logger.debug(f"V4: Selected {chunk.rel_path} (sim={score:.3f}, {tokens} tokens)")
        
        # Calculate metrics
        coverage_score = len(selected) / max(1, len(chunks))
        diversity_score = len(set(chunk.rel_path for chunk in selected)) / max(1, len(selected))
        
        # Ensure deterministic ordering if requested
        if config.deterministic and selected:
            # Sort by similarity score (descending) then by path for deterministic ordering
            selected.sort(key=lambda c: (
                -selection_scores.get(c.id, 0.0),  # Highest similarity first
                c.rel_path,  # Then by path
                c.id  # Finally by chunk ID
            ))
        
        # Create result
        result = SelectionResult(
            selected_chunks=selected,
            chunk_modes=chunk_modes,
            selection_scores=selection_scores,
            total_tokens=total_tokens,
            budget_utilization=total_tokens / config.token_budget,
            coverage_score=coverage_score,
            diversity_score=diversity_score,
            iterations=iterations,
        )
        
        avg_similarity = sum(selection_scores.values()) / len(selection_scores) if selection_scores else 0.0
        logger.info(
            f"V4 completed: {len(selected)} chunks, {total_tokens} tokens, "
            f"{result.budget_utilization:.1%} budget utilization, "
            f"avg similarity: {avg_similarity:.3f}"
        )
        
        return result
    
    def _validate_baseline_constraints(self, result: SelectionResult, config: BaselineConfig) -> List[str]:
        """Validate V4-specific constraints."""
        errors = []
        
        # V4 should show semantic ranking bias - check that scores are generally decreasing
        if len(result.selection_scores) >= 2:
            scores = list(result.selection_scores.values())
            
            # Check if scores are in reasonable descending order
            ascending_pairs = 0
            descending_pairs = 0
            
            for i in range(len(scores) - 1):
                if scores[i] > scores[i + 1]:
                    descending_pairs += 1
                elif scores[i] < scores[i + 1]:
                    ascending_pairs += 1
            
            if descending_pairs + ascending_pairs > 0:
                descending_ratio = descending_pairs / (descending_pairs + ascending_pairs)
                # Expect at least 50% descending order for semantic ranking
                if descending_ratio < 0.5:
                    errors.append(
                        f"V4 selection doesn't show semantic ranking bias: "
                        f"only {descending_ratio:.1%} of adjacent pairs show similarity decrease"
                    )
        
        # Check that we computed embeddings
        if not self._chunk_embeddings:
            errors.append("V4 failed to compute any embeddings")
        
        return errors
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get V4-specific performance metrics."""
        base_metrics = super().get_performance_metrics()
        base_metrics.update({
            "selection_strategy": "semantic_similarity",
            "supports_signature_fallback": True,
            "embedding_model": self.model_name,
            "computed_embeddings": len(self._chunk_embeddings),
            "using_fallback": self._embedding_model == "fallback",
            "theoretical_bias": "semantic_relevance",
        })
        return base_metrics
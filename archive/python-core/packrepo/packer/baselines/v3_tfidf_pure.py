"""V3: Enhanced Pure TF-IDF Baseline Implementation."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Set

from .base import BaselineSelector, BaselineConfig
from ..chunker.base import Chunk
from ..selector.base import SelectionResult

logger = logging.getLogger(__name__)


class V3TfIdfPureBaseline(BaselineSelector):
    """
    V3: Enhanced Pure TF-IDF Baseline - Improve existing BM25 implementation.
    
    Pure TF-IDF ranking without BM25 modifications using standard TF-IDF scoring
    with document-term matrix. This provides a strong information retrieval
    baseline that uses classic term frequency and inverse document frequency
    calculations.
    
    Unlike the V0c BM25 baseline, this uses pure TF-IDF without length normalization
    or saturation parameters, providing a different retrieval characteristic.
    """
    
    def __init__(self, tokenizer=None):
        """Initialize TF-IDF baseline with text processing capabilities."""
        super().__init__(tokenizer)
        self._term_cache: Dict[str, Dict[str, float]] = {}  # chunk_id -> {term: tf}
        self._document_frequencies: Dict[str, int] = {}  # term -> document count
        self._vocabulary: Set[str] = set()
    
    def get_variant_id(self) -> str:
        """Get the baseline variant identifier."""
        return "V3"
    
    def get_description(self) -> str:
        """Get human-readable description of baseline."""
        return "Pure TF-IDF information retrieval baseline"
    
    def _extract_terms(self, text: str) -> List[str]:
        """
        Extract and normalize terms from text for TF-IDF calculation.
        
        Args:
            text: Input text to process
            
        Returns:
            List of normalized terms
        """
        if not text:
            return []
        
        # Convert to lowercase and split on non-alphanumeric characters
        # Keep underscores and preserve programming identifiers
        text = text.lower()
        
        # Extract terms: alphanumeric sequences including underscores
        terms = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)
        
        # Filter out very short terms and common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            # Programming common terms that might not be useful
            'var', 'let', 'const', 'if', 'else', 'for', 'while', 'do', 'try', 'catch', 'throw',
        }
        
        filtered_terms = []
        for term in terms:
            if len(term) >= 2 and term not in stop_words:
                filtered_terms.append(term)
        
        return filtered_terms
    
    def _calculate_tf(self, terms: List[str]) -> Dict[str, float]:
        """
        Calculate term frequency scores for a document.
        
        Args:
            terms: List of terms in the document
            
        Returns:
            Dictionary mapping terms to TF scores
        """
        if not terms:
            return {}
        
        term_counts = Counter(terms)
        total_terms = len(terms)
        
        tf_scores = {}
        for term, count in term_counts.items():
            # Use log normalization: 1 + log(count)
            tf_scores[term] = 1.0 + math.log(count)
        
        return tf_scores
    
    def _calculate_idf(self, term: str, total_documents: int) -> float:
        """
        Calculate inverse document frequency for a term.
        
        Args:
            term: The term to calculate IDF for
            total_documents: Total number of documents in corpus
            
        Returns:
            IDF score for the term
        """
        doc_frequency = self._document_frequencies.get(term, 0)
        if doc_frequency == 0:
            return 0.0
        
        # Standard IDF: log(N / df)
        return math.log(total_documents / doc_frequency)
    
    def _build_document_index(self, chunks: List[Chunk]) -> None:
        """
        Build the document-term index for TF-IDF calculation.
        
        Args:
            chunks: All available chunks to index
        """
        logger.info(f"V3: Building TF-IDF index for {len(chunks)} chunks")
        
        self._term_cache.clear()
        self._document_frequencies.clear()
        self._vocabulary.clear()
        
        # Extract terms from each chunk
        for chunk in chunks:
            # Combine content, docstring, and signature for term extraction
            text_parts = []
            if chunk.content:
                text_parts.append(chunk.content)
            if chunk.docstring:
                text_parts.append(chunk.docstring)
            if chunk.signature:
                text_parts.append(chunk.signature)
            # Also include file path and chunk name for context
            text_parts.extend([chunk.rel_path, chunk.name])
            
            full_text = ' '.join(text_parts)
            terms = self._extract_terms(full_text)
            
            if terms:
                # Calculate TF scores for this chunk
                tf_scores = self._calculate_tf(terms)
                self._term_cache[chunk.id] = tf_scores
                
                # Update document frequencies
                unique_terms = set(terms)
                for term in unique_terms:
                    self._document_frequencies[term] = self._document_frequencies.get(term, 0) + 1
                    self._vocabulary.add(term)
        
        logger.info(f"V3: Index built with {len(self._vocabulary)} unique terms")
    
    def _calculate_tfidf_scores(self, chunks: List[Chunk], query_terms: List[str]) -> Dict[str, float]:
        """
        Calculate TF-IDF scores for chunks against query terms.
        
        Args:
            chunks: Chunks to score
            query_terms: Terms to score against
            
        Returns:
            Dictionary mapping chunk IDs to TF-IDF scores
        """
        scores = {}
        total_documents = len(chunks)
        
        for chunk in chunks:
            tf_scores = self._term_cache.get(chunk.id, {})
            tfidf_score = 0.0
            
            for term in query_terms:
                if term in tf_scores:
                    tf = tf_scores[term]
                    idf = self._calculate_idf(term, total_documents)
                    tfidf_score += tf * idf
            
            scores[chunk.id] = tfidf_score
        
        return scores
    
    def _generate_query_from_repository(self, chunks: List[Chunk]) -> List[str]:
        """
        Generate a query from repository characteristics for TF-IDF scoring.
        
        Since we don't have an explicit query, we'll use the most important
        terms from the repository as a whole to score relevance.
        
        Args:
            chunks: All available chunks
            
        Returns:
            List of query terms representing important repository concepts
        """
        # Collect all terms with their frequencies
        term_frequencies = Counter()
        
        for chunk_id, tf_scores in self._term_cache.items():
            for term, tf_score in tf_scores.items():
                term_frequencies[term] += 1  # Document frequency
        
        # Calculate importance scores: use terms that appear in multiple documents
        # but not too frequently (to avoid common/generic terms)
        total_docs = len(self._term_cache)
        important_terms = []
        
        for term, doc_freq in term_frequencies.items():
            # Terms that appear in 5-50% of documents are usually most informative
            doc_ratio = doc_freq / total_docs
            if 0.05 <= doc_ratio <= 0.5 and len(term) >= 3:
                # Score by inverse document frequency (rarer = more important)
                importance = math.log(total_docs / doc_freq)
                important_terms.append((term, importance))
        
        # Sort by importance and take top terms
        important_terms.sort(key=lambda x: -x[1])
        query_terms = [term for term, _ in important_terms[:50]]  # Top 50 terms
        
        logger.info(f"V3: Generated query with {len(query_terms)} terms: {query_terms[:10]}...")
        return query_terms
    
    def _select_baseline(self, chunks: List[Chunk], config: BaselineConfig) -> SelectionResult:
        """
        Select chunks using pure TF-IDF scoring.
        
        Selection strategy:
        1. Build TF-IDF index for all chunks
        2. Generate repository-wide query from important terms
        3. Score all chunks using TF-IDF against query
        4. Select highest-scoring chunks until budget exhausted
        
        Args:
            chunks: Preprocessed chunks to select from
            config: Baseline configuration
            
        Returns:
            Selection result with TF-IDF ranked chunks
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
        
        logger.info(f"V3: Starting TF-IDF selection from {len(chunks)} chunks")
        
        # Build TF-IDF index
        self._build_document_index(chunks)
        
        # Generate query from repository characteristics
        query_terms = self._generate_query_from_repository(chunks)
        
        if not query_terms:
            # Fallback: use file names and common programming terms
            query_terms = ['main', 'class', 'function', 'method', 'init', 'test', 'util', 'helper']
            logger.warning("V3: No repository-specific terms found, using fallback query")
        
        # Calculate TF-IDF scores for all chunks
        tfidf_scores = self._calculate_tfidf_scores(chunks, query_terms)
        
        # Sort chunks by TF-IDF score (descending)
        sorted_chunks = sorted(chunks, key=lambda c: tfidf_scores.get(c.id, 0.0), reverse=True)
        
        # Log top scores for debugging
        top_scores = [(c.rel_path, tfidf_scores.get(c.id, 0.0)) for c in sorted_chunks[:5]]
        logger.info(f"V3: Top TF-IDF scores: {top_scores}")
        
        # Select chunks in TF-IDF score order until budget exhausted
        for chunk in sorted_chunks:
            if total_tokens >= config.token_budget:
                break
                
            iterations += 1
            score = tfidf_scores.get(chunk.id, 0.0)
            
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
                    
                    logger.debug(f"V3: Using signature mode for {chunk.rel_path} (TF-IDF={score:.3f})")
                else:
                    # Skip if even signature doesn't fit
                    logger.debug(f"V3: Skipping {chunk.rel_path} - too large even in signature mode")
                continue
            
            # Add chunk in full mode
            selected.append(chunk)
            selected_ids.add(chunk.id)
            chunk_modes[chunk.id] = "full"
            selection_scores[chunk.id] = score
            total_tokens += tokens
            
            logger.debug(f"V3: Selected {chunk.rel_path} (TF-IDF={score:.3f}, {tokens} tokens)")
        
        # Calculate metrics
        coverage_score = len(selected) / max(1, len(chunks))
        diversity_score = len(set(chunk.rel_path for chunk in selected)) / max(1, len(selected))
        
        # Ensure deterministic ordering if requested
        if config.deterministic and selected:
            # Sort by TF-IDF score (descending) then by path for deterministic ordering
            selected.sort(key=lambda c: (
                -selection_scores.get(c.id, 0.0),  # Highest score first
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
        
        logger.info(
            f"V3 completed: {len(selected)} chunks, {total_tokens} tokens, "
            f"{result.budget_utilization:.1%} budget utilization, "
            f"avg TF-IDF score: {sum(selection_scores.values()) / len(selection_scores):.3f}"
        )
        
        return result
    
    def _validate_baseline_constraints(self, result: SelectionResult, config: BaselineConfig) -> List[str]:
        """Validate V3-specific constraints."""
        errors = []
        
        # V3 should show TF-IDF ranking bias - check that scores are generally decreasing
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
                # Expect at least 40% descending order (allowing for ties and similar scores)
                if descending_ratio < 0.4:
                    errors.append(
                        f"V3 selection doesn't show TF-IDF ranking bias: "
                        f"only {descending_ratio:.1%} of adjacent pairs show score decrease"
                    )
        
        # Check that we have reasonable term coverage
        if len(self._vocabulary) < 10:
            errors.append(f"V3 extracted very few terms ({len(self._vocabulary)})")
        
        return errors
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get V3-specific performance metrics."""
        base_metrics = super().get_performance_metrics()
        base_metrics.update({
            "selection_strategy": "pure_tfidf",
            "supports_signature_fallback": True,
            "vocabulary_size": len(self._vocabulary),
            "indexed_documents": len(self._term_cache),
            "theoretical_bias": "term_frequency_relevance",
        })
        return base_metrics
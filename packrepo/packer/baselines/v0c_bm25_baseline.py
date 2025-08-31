"""V0c: BM25 Baseline Implementation with TF-IDF Ranking."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Set, Optional, Tuple
from pathlib import Path

from .base import BaselineSelector, BaselineConfig
from ..chunker.base import Chunk
from ..selector.base import SelectionResult

logger = logging.getLogger(__name__)


class V0cBM25Baseline(BaselineSelector):
    """
    V0c: BM25 Baseline - Traditional information retrieval ranking.
    
    Implements BM25 (Best Matching 25) scoring with TF-IDF features
    to create a strong traditional baseline for comparison against
    sophisticated variants.
    
    This represents the state-of-the-art in traditional IR before
    neural approaches, providing a challenging baseline to beat.
    
    Features:
    - BM25 scoring with tuned parameters
    - TF-IDF document representation
    - Code-aware tokenization (identifiers, keywords)
    - File type boosting (README, main files)
    - Deterministic ranking with tie-breakers
    """
    
    def __init__(self, tokenizer=None):
        """Initialize BM25 baseline with IR parameters."""
        super().__init__(tokenizer)
        
        # BM25 parameters (tuned for code repositories)
        self.k1 = 1.5  # Term frequency saturation parameter
        self.b = 0.75  # Document length normalization parameter
        
        # TF-IDF parameters
        self.min_df = 1    # Minimum document frequency
        self.max_df = 0.95 # Maximum document frequency (ignore very common terms)
        
        # Code-specific parameters
        self.boost_identifiers = 1.2    # Boost for programming identifiers
        self.boost_keywords = 1.1       # Boost for language keywords
        self.boost_comments = 0.9       # Slight boost for comments/docs
        
        # File type boosting
        self.file_type_boosts = {
            'readme': 2.0,
            'main': 1.8,
            'index': 1.6,
            'config': 1.4,
            'test': 0.8,
            'generated': 0.5,
        }
        
        # Internal state
        self._vocabulary: Dict[str, int] = {}
        self._document_frequencies: Dict[str, int] = {}
        self._avg_doc_length: float = 0.0
        self._doc_lengths: Dict[str, int] = {}
        
    def get_variant_id(self) -> str:
        """Get the baseline variant identifier."""
        return "V0c"
    
    def get_description(self) -> str:
        """Get human-readable description of baseline."""
        return "BM25 + TF-IDF traditional IR baseline"
    
    def _select_baseline(self, chunks: List[Chunk], config: BaselineConfig) -> SelectionResult:
        """
        Select chunks using BM25 scoring with TF-IDF features.
        
        Process:
        1. Extract and tokenize chunk content
        2. Build vocabulary and document frequencies
        3. Calculate BM25 scores for all chunks
        4. Apply file type and quality boosts
        5. Rank chunks and select top-N within budget
        
        Args:
            chunks: Preprocessed chunks to select from
            config: Baseline configuration
            
        Returns:
            Selection result with BM25-ranked chunks
        """
        logger.info(f"V0c: Starting BM25 baseline with {len(chunks)} chunks")
        
        # Step 1: Build corpus and vocabulary
        self._build_corpus_statistics(chunks)
        
        # Step 2: Calculate BM25 scores for all chunks
        scored_chunks = self._calculate_bm25_scores(chunks)
        
        # Step 3: Apply additional boosts
        scored_chunks = self._apply_quality_boosts(scored_chunks)
        
        # Step 4: Sort by score (descending) with deterministic tie-breaking
        if config.deterministic:
            scored_chunks.sort(key=lambda x: (-x[1], x[0].rel_path, x[0].start_line, x[0].id))
        else:
            scored_chunks.sort(key=lambda x: -x[1])
        
        # Step 5: Select top chunks within budget
        selected = []
        selected_ids = set()
        chunk_modes = {}
        selection_scores = {}
        total_tokens = 0
        iterations = 0
        
        for chunk, score in scored_chunks:
            if total_tokens >= config.token_budget:
                break
                
            iterations += 1
            
            # Try full mode first
            full_tokens = self._get_chunk_tokens(chunk, "full")
            if total_tokens + full_tokens <= config.token_budget:
                selected.append(chunk)
                selected_ids.add(chunk.id)
                chunk_modes[chunk.id] = "full"
                selection_scores[chunk.id] = score
                total_tokens += full_tokens
                
                logger.debug(f"V0c: Selected {chunk.rel_path}:{chunk.start_line} (score={score:.3f}, full mode)")
                continue
            
            # Try signature mode
            sig_tokens = self._get_chunk_tokens(chunk, "signature")
            if total_tokens + sig_tokens <= config.token_budget:
                selected.append(chunk)
                selected_ids.add(chunk.id)
                chunk_modes[chunk.id] = "signature"
                selection_scores[chunk.id] = score
                total_tokens += sig_tokens
                
                logger.debug(f"V0c: Selected {chunk.rel_path}:{chunk.start_line} (score={score:.3f}, sig mode)")
                continue
            
            # Skip if even signature doesn't fit
            logger.debug(f"V0c: Skipping {chunk.rel_path}:{chunk.start_line} - too large")
        
        # Calculate quality metrics
        coverage_score = len(selected) / max(1, len(chunks))
        diversity_score = self._calculate_diversity_score(selected)
        
        # Ensure deterministic ordering in result
        if config.deterministic and selected:
            selected.sort(key=lambda c: (c.rel_path, c.start_line, c.id))
        
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
            f"V0c completed: {len(selected)} chunks, {total_tokens} tokens, "
            f"{result.budget_utilization:.1%} budget utilization, "
            f"avg_score={sum(result.selection_scores.values()) / max(1, len(result.selection_scores)):.3f}"
        )
        
        return result
    
    def _build_corpus_statistics(self, chunks: List[Chunk]):
        """Build vocabulary and document frequency statistics."""
        logger.info("V0c: Building corpus statistics")
        
        # Reset internal state
        self._vocabulary.clear()
        self._document_frequencies.clear()
        self._doc_lengths.clear()
        
        # Collect all terms and document frequencies
        all_doc_terms = []
        vocab_counter = Counter()
        
        for chunk in chunks:
            # Tokenize chunk content
            terms = self._tokenize_chunk(chunk)
            doc_terms = list(terms)
            all_doc_terms.append(doc_terms)
            
            # Update vocabulary
            vocab_counter.update(terms)
            
            # Update document frequencies (unique terms per document)
            unique_terms = set(terms)
            for term in unique_terms:
                self._document_frequencies[term] = self._document_frequencies.get(term, 0) + 1
            
            # Store document length
            self._doc_lengths[chunk.id] = len(doc_terms)
        
        # Build vocabulary (filter by document frequency)
        total_docs = len(chunks)
        self._vocabulary = {}
        vocab_id = 0
        
        for term, count in vocab_counter.items():
            # Apply document frequency filters
            df = self._document_frequencies[term]
            if df >= self.min_df and df <= (self.max_df * total_docs):
                self._vocabulary[term] = vocab_id
                vocab_id += 1
        
        # Calculate average document length
        if self._doc_lengths:
            self._avg_doc_length = sum(self._doc_lengths.values()) / len(self._doc_lengths)
        else:
            self._avg_doc_length = 1.0
        
        logger.info(
            f"V0c: Built vocabulary with {len(self._vocabulary)} terms, "
            f"avg_doc_length={self._avg_doc_length:.1f}"
        )
    
    def _tokenize_chunk(self, chunk: Chunk) -> List[str]:
        """
        Tokenize chunk content with code-aware processing.
        
        Extracts:
        - Programming identifiers (variables, functions, classes)
        - Keywords and reserved words
        - Comments and documentation
        - String literals (normalized)
        - File path components
        """
        terms = []
        content = chunk.content or ""
        
        # Add file path components as high-value terms
        path_parts = chunk.rel_path.replace('/', ' ').replace('_', ' ').replace('-', ' ').split()
        path_terms = [part.lower() for part in path_parts if len(part) > 1]
        terms.extend(path_terms)
        
        # Extract programming identifiers (camelCase, snake_case, etc.)
        identifier_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        identifiers = re.findall(identifier_pattern, content)
        for identifier in identifiers:
            if len(identifier) > 1:  # Skip single-character identifiers
                # Split camelCase and snake_case
                split_terms = self._split_identifier(identifier.lower())
                terms.extend(split_terms)
        
        # Extract comments and documentation
        comment_patterns = [
            r'//\s*(.+)',        # // comments
            r'/\*\s*(.+?)\s*\*/',  # /* */ comments
            r'#\s*(.+)',          # # comments
            r'"""\s*(.+?)\s*"""', # """ docstrings
            r"'''\s*(.+?)\s*'''", # ''' docstrings
        ]
        
        for pattern in comment_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                comment_terms = self._extract_words(match.lower())
                terms.extend(comment_terms)
        
        # Extract string literals (normalized)
        string_patterns = [
            r'"([^"]*)"',  # Double quotes
            r"'([^']*)'",  # Single quotes
        ]
        
        for pattern in string_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if len(match) > 2:  # Skip very short strings
                    string_terms = self._extract_words(match.lower())
                    terms.extend(string_terms)
        
        # Extract general words from content
        general_words = self._extract_words(content.lower())
        terms.extend(general_words)
        
        # Filter and return unique terms
        filtered_terms = [term for term in terms if len(term) > 1 and term.isalpha()]
        return filtered_terms
    
    def _split_identifier(self, identifier: str) -> List[str]:
        """Split camelCase and snake_case identifiers into components."""
        # Handle snake_case
        parts = identifier.split('_')
        
        # Handle camelCase
        result = []
        for part in parts:
            if part:
                # Split on capital letters
                camel_parts = re.findall(r'[a-z]+|[A-Z][a-z]*', part)
                result.extend([p.lower() for p in camel_parts if p])
        
        return result
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract alphabetic words from text."""
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        return [word.lower() for word in words]
    
    def _calculate_bm25_scores(self, chunks: List[Chunk]) -> List[Tuple[Chunk, float]]:
        """Calculate BM25 scores for all chunks."""
        logger.info("V0c: Calculating BM25 scores")
        
        scored_chunks = []
        total_docs = len(chunks)
        
        for chunk in chunks:
            # Tokenize chunk
            terms = self._tokenize_chunk(chunk)
            
            # Calculate term frequencies
            term_frequencies = Counter(terms)
            doc_length = len(terms)
            
            # Calculate BM25 score
            score = 0.0
            
            for term, tf in term_frequencies.items():
                if term not in self._vocabulary:
                    continue  # Skip out-of-vocabulary terms
                
                # Document frequency
                df = self._document_frequencies.get(term, 1)
                
                # IDF component
                idf = math.log((total_docs - df + 0.5) / (df + 0.5))
                
                # TF component with BM25 normalization
                tf_component = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * (doc_length / self._avg_doc_length))
                )
                
                # Add to score
                score += idf * tf_component
            
            # Apply term-specific boosts
            score = self._apply_term_boosts(score, terms)
            
            scored_chunks.append((chunk, score))
        
        logger.info(f"V0c: Calculated BM25 scores for {len(scored_chunks)} chunks")
        return scored_chunks
    
    def _apply_term_boosts(self, base_score: float, terms: List[str]) -> float:
        """Apply term-specific boosts to base BM25 score."""
        # Define programming keywords (common across languages)
        programming_keywords = {
            'class', 'function', 'def', 'var', 'let', 'const', 'import', 'export',
            'public', 'private', 'protected', 'static', 'async', 'await',
            'return', 'if', 'else', 'for', 'while', 'try', 'catch', 'throw',
        }
        
        # Define identifier patterns
        identifier_patterns = {
            'api', 'service', 'controller', 'model', 'view', 'component',
            'handler', 'manager', 'factory', 'builder', 'config', 'util',
        }
        
        # Define documentation terms
        doc_terms = {
            'readme', 'documentation', 'doc', 'guide', 'tutorial', 'example',
            'description', 'overview', 'introduction', 'usage', 'install',
        }
        
        boosted_score = base_score
        term_set = set(terms)
        
        # Boost for programming keywords
        keyword_count = len(term_set.intersection(programming_keywords))
        if keyword_count > 0:
            boosted_score *= (1 + keyword_count * 0.1 * self.boost_keywords)
        
        # Boost for common identifiers
        identifier_count = len(term_set.intersection(identifier_patterns))
        if identifier_count > 0:
            boosted_score *= (1 + identifier_count * 0.1 * self.boost_identifiers)
        
        # Boost for documentation terms
        doc_count = len(term_set.intersection(doc_terms))
        if doc_count > 0:
            boosted_score *= (1 + doc_count * 0.1 * self.boost_comments)
        
        return boosted_score
    
    def _apply_quality_boosts(self, scored_chunks: List[Tuple[Chunk, float]]) -> List[Tuple[Chunk, float]]:
        """Apply file-type and quality-based score boosts."""
        logger.info("V0c: Applying quality boosts")
        
        boosted_chunks = []
        
        for chunk, score in scored_chunks:
            boosted_score = score
            file_path_lower = chunk.rel_path.lower()
            file_name = Path(chunk.rel_path).name.lower()
            
            # File type boosts
            if 'readme' in file_name:
                boosted_score *= self.file_type_boosts['readme']
            elif any(name in file_name for name in ['main', 'index', '__init__']):
                boosted_score *= self.file_type_boosts['main']
            elif 'index' in file_name:
                boosted_score *= self.file_type_boosts['index']
            elif any(name in file_name for name in ['config', 'settings', 'conf']):
                boosted_score *= self.file_type_boosts['config']
            elif 'test' in file_path_lower:
                boosted_score *= self.file_type_boosts['test']
            elif any(marker in file_path_lower for marker in ['generated', 'auto', 'build', 'dist']):
                boosted_score *= self.file_type_boosts['generated']
            
            # Documentation density boost
            if hasattr(chunk, 'doc_density') and chunk.doc_density > 0:
                boosted_score *= (1 + chunk.doc_density * 0.3)
            
            # Complexity penalty (prefer simpler code for overview)
            if hasattr(chunk, 'complexity_score') and chunk.complexity_score > 5.0:
                complexity_penalty = 1.0 - min(0.3, (chunk.complexity_score - 5.0) / 20.0)
                boosted_score *= complexity_penalty
            
            boosted_chunks.append((chunk, boosted_score))
        
        return boosted_chunks
    
    def _calculate_diversity_score(self, selected: List[Chunk]) -> float:
        """Calculate diversity score based on file and term coverage."""
        if not selected:
            return 0.0
        
        # File diversity
        unique_files = set(chunk.rel_path for chunk in selected)
        file_diversity = len(unique_files) / len(selected)
        
        # Term diversity (approximate)
        all_terms = set()
        for chunk in selected:
            terms = self._tokenize_chunk(chunk)
            all_terms.update(terms)
        
        term_diversity = len(all_terms) / max(1, sum(len(self._tokenize_chunk(chunk)) for chunk in selected))
        
        # Combined diversity score
        diversity = (file_diversity + term_diversity) / 2.0
        return min(1.0, diversity)
    
    def _validate_baseline_constraints(self, result: SelectionResult, config: BaselineConfig) -> List[str]:
        """Validate V0c-specific constraints."""
        errors = []
        
        # V0c should have monotonic decreasing scores
        if result.selection_scores and not config.deterministic:
            scores = [result.selection_scores.get(chunk.id, 0) for chunk in result.selected_chunks]
            for i in range(1, len(scores)):
                if scores[i] > scores[i-1] + 1e-10:  # Allow for floating point precision
                    errors.append(f"V0c selection scores not monotonic at position {i}")
                    break
        
        # V0c should have reasonable vocabulary utilization
        if hasattr(self, '_vocabulary') and len(self._vocabulary) > 0:
            # Check that selection uses diverse vocabulary
            selected_terms = set()
            for chunk in result.selected_chunks:
                terms = self._tokenize_chunk(chunk)
                selected_terms.update(term for term in terms if term in self._vocabulary)
            
            vocab_utilization = len(selected_terms) / len(self._vocabulary)
            if vocab_utilization < 0.1 and len(result.selected_chunks) > 10:
                errors.append(f"V0c low vocabulary utilization: {vocab_utilization:.1%}")
        
        return errors
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get V0c-specific performance metrics."""
        base_metrics = super().get_performance_metrics()
        base_metrics.update({
            "selection_strategy": "bm25_tfidf_ranking",
            "supports_signature_fallback": True,
            "vocabulary_size": len(self._vocabulary),
            "avg_doc_length": self._avg_doc_length,
            "bm25_k1": self.k1,
            "bm25_b": self.b,
            "code_aware_tokenization": True,
            "file_type_boosting": True,
        })
        return base_metrics
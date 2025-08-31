"""
Fast facility location approximation for submodular file selection.

Implements a lightweight version of submodular optimization using TF-IDF
features and greedy facility location for diverse file selection.
Designed for <3s execution on typical repositories.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..fastpath.fast_scan import ScanResult
from ..fastpath.heuristics import ScoreComponents
from ..packer.tokenizer import estimate_tokens_scan_result


@dataclass 
class SelectionResult:
    """Result of submodular file selection."""
    selected_files: List[ScanResult]
    selection_scores: List[float]
    diversity_score: float
    coverage_score: float
    total_tokens: int
    selection_time: float


class TFIDFFeatureExtractor:
    """Lightweight TF-IDF feature extraction for file diversity."""
    
    def __init__(self, min_token_freq: int = 2, max_features: int = 1000):
        self.min_token_freq = min_token_freq
        self.max_features = max_features
        
        # Vocabulary and document frequency
        self.vocab: Dict[str, int] = {}
        self.doc_freq: Counter = Counter()
        self.idf_scores: Dict[str, float] = {}
        
    def _tokenize_content(self, result: ScanResult) -> List[str]:
        """Extract tokens from file for TF-IDF analysis."""
        tokens = []
        
        # Use file path components
        path_parts = result.stats.path.lower().replace('/', ' ').replace('\\', ' ').split()
        tokens.extend(path_parts)
        
        # Use file language
        if result.stats.language:
            tokens.append(result.stats.language)
            
        # Use import information
        if result.imports and result.imports.imports:
            for imp in result.imports.imports:
                # Split import paths into tokens
                import_tokens = imp.lower().replace('.', ' ').replace('/', ' ').split()
                tokens.extend(import_tokens)
                
        # Use document analysis
        if result.doc_analysis:
            # Add synthetic tokens based on document structure
            if result.doc_analysis.heading_count > 0:
                tokens.append('has_headings')
            if result.doc_analysis.toc_indicators > 0:
                tokens.append('has_toc')
            if result.doc_analysis.link_count > 0:
                tokens.append('has_links')
            if result.doc_analysis.code_block_count > 0:
                tokens.append('has_code_blocks')
                
        # Add file type indicators
        if result.stats.is_readme:
            tokens.append('readme_file')
        if result.stats.is_test:
            tokens.append('test_file')
        if result.stats.is_config:
            tokens.append('config_file')
        if result.stats.is_docs:
            tokens.append('docs_file')
            
        # Filter out short tokens and common words
        filtered_tokens = []
        stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        for token in tokens:
            if len(token) >= 2 and token not in stopwords:
                filtered_tokens.append(token)
                
        return filtered_tokens
        
    def fit(self, scan_results: List[ScanResult]) -> None:
        """Build vocabulary and compute IDF scores."""
        # Count document frequencies
        doc_count = len(scan_results)
        
        for result in scan_results:
            tokens = self._tokenize_content(result)
            unique_tokens = set(tokens)
            
            for token in unique_tokens:
                self.doc_freq[token] += 1
                
        # Filter vocabulary by frequency and limit size
        filtered_vocab = {
            token: freq for token, freq in self.doc_freq.items()
            if freq >= self.min_token_freq
        }
        
        # Sort by frequency and take top features
        sorted_terms = sorted(filtered_vocab.items(), key=lambda x: x[1], reverse=True)
        top_terms = sorted_terms[:self.max_features]
        
        # Build vocabulary index
        self.vocab = {token: idx for idx, (token, freq) in enumerate(top_terms)}
        
        # Compute IDF scores
        for token in self.vocab:
            df = self.doc_freq[token]
            idf = math.log(doc_count / df)
            self.idf_scores[token] = idf
            
    def transform(self, result: ScanResult) -> Dict[int, float]:
        """Transform a scan result into TF-IDF feature vector (sparse)."""
        tokens = self._tokenize_content(result)
        token_counts = Counter(tokens)
        
        # Compute TF-IDF features
        features = {}
        total_tokens = len(tokens)
        
        for token, count in token_counts.items():
            if token in self.vocab:
                feature_idx = self.vocab[token]
                tf = count / max(total_tokens, 1)
                idf = self.idf_scores[token]
                tfidf = tf * idf
                
                if tfidf > 0.0:
                    features[feature_idx] = tfidf
                    
        return features
        
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1])
        return [token for token, idx in sorted_vocab]


class FastFacilityLocation:
    """
    Fast facility location for diverse file selection.
    
    Uses submodular optimization principles with TF-IDF features
    to select diverse, high-quality files within token budget.
    """
    
    def __init__(self, diversity_weight: float = 0.3, coverage_weight: float = 0.7):
        self.diversity_weight = diversity_weight
        self.coverage_weight = coverage_weight
        
        # Feature extractor
        self.feature_extractor = TFIDFFeatureExtractor()
        self.file_features: Dict[int, Dict[int, float]] = {}
        
    def _compute_similarity(self, features1: Dict[int, float], features2: Dict[int, float]) -> float:
        """Compute cosine similarity between two sparse feature vectors."""
        # Find common features
        common_features = set(features1.keys()) & set(features2.keys())
        
        if not common_features:
            return 0.0
            
        # Compute dot product and norms
        dot_product = sum(features1[feat] * features2[feat] for feat in common_features)
        
        norm1 = math.sqrt(sum(val * val for val in features1.values()))
        norm2 = math.sqrt(sum(val * val for val in features2.values()))
        
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
        
    def _compute_marginal_gain(self, candidate_idx: int, selected_indices: Set[int],
                              scored_files: List[Tuple[ScanResult, ScoreComponents]]) -> float:
        """Compute marginal gain of adding candidate to selection."""
        candidate_result, candidate_score = scored_files[candidate_idx]
        candidate_features = self.file_features[candidate_idx]
        
        # Quality component (normalized score)
        quality_gain = candidate_score.final_score
        
        # Diversity component (based on dissimilarity to selected files)
        if not selected_indices:
            diversity_gain = 1.0  # First file gets maximum diversity
        else:
            # Compute max similarity to selected files
            max_similarity = 0.0
            for selected_idx in selected_indices:
                selected_features = self.file_features[selected_idx]
                similarity = self._compute_similarity(candidate_features, selected_features)
                max_similarity = max(max_similarity, similarity)
                
            # Diversity gain is inverse of max similarity
            diversity_gain = 1.0 - max_similarity
            
        # Combined marginal gain
        marginal_gain = (self.coverage_weight * quality_gain + 
                        self.diversity_weight * diversity_gain)
        
        return marginal_gain
        
    def _estimate_tokens(self, result: ScanResult) -> int:
        """Estimate token count for a file using centralized utility."""
        return estimate_tokens_scan_result(result, use_lines=True)
        
    def select_files(self, scored_files: List[Tuple[ScanResult, ScoreComponents]], 
                    token_budget: int) -> SelectionResult:
        """
        Select diverse set of files within token budget.
        
        Uses greedy submodular optimization with facility location objective.
        """
        import time
        start_time = time.time()
        
        if not scored_files:
            return SelectionResult([], [], 0.0, 0.0, 0, time.time() - start_time)
            
        # Extract scan results for feature extraction
        scan_results = [result for result, score in scored_files]
        
        # Fit TF-IDF features
        self.feature_extractor.fit(scan_results)
        
        # Transform all files to feature space
        for idx, (result, score) in enumerate(scored_files):
            self.file_features[idx] = self.feature_extractor.transform(result)
            
        # Greedy selection
        selected_indices = set()
        selected_files = []
        selection_scores = []
        total_tokens = 0
        
        # Must-include README files first (if within budget)
        readme_indices = [idx for idx, (result, score) in enumerate(scored_files) 
                         if result.stats.is_readme]
        
        for idx in readme_indices:
            result, score = scored_files[idx]
            estimated_tokens = self._estimate_tokens(result)
            
            if total_tokens + estimated_tokens <= token_budget:
                selected_indices.add(idx)
                selected_files.append(result)
                selection_scores.append(score.final_score)
                total_tokens += estimated_tokens
                
        # Greedy selection for remaining budget
        candidates = [idx for idx in range(len(scored_files)) if idx not in selected_indices]
        
        while candidates and total_tokens < token_budget:
            best_idx = None
            best_gain = -1.0
            best_tokens = 0
            
            # Find candidate with best marginal gain per token
            for idx in candidates:
                result, score = scored_files[idx]
                estimated_tokens = self._estimate_tokens(result)
                
                # Skip if doesn't fit in budget
                if total_tokens + estimated_tokens > token_budget:
                    continue
                    
                marginal_gain = self._compute_marginal_gain(idx, selected_indices, scored_files)
                
                # Normalize by token cost
                gain_per_token = marginal_gain / max(estimated_tokens, 1)
                
                if gain_per_token > best_gain:
                    best_gain = gain_per_token
                    best_idx = idx
                    best_tokens = estimated_tokens
                    
            # Add best candidate
            if best_idx is not None:
                selected_indices.add(best_idx)
                result, score = scored_files[best_idx]
                selected_files.append(result)
                selection_scores.append(score.final_score)
                total_tokens += best_tokens
                candidates.remove(best_idx)
            else:
                # No more candidates fit in budget
                break
                
        # Compute final scores
        diversity_score = self._compute_diversity_score(selected_indices)
        coverage_score = sum(selection_scores) / len(selection_scores) if selection_scores else 0.0
        
        selection_time = time.time() - start_time
        
        return SelectionResult(
            selected_files=selected_files,
            selection_scores=selection_scores,
            diversity_score=diversity_score,
            coverage_score=coverage_score,
            total_tokens=total_tokens,
            selection_time=selection_time
        )
        
    def _compute_diversity_score(self, selected_indices: Set[int]) -> float:
        """Compute overall diversity score of selection."""
        if len(selected_indices) <= 1:
            return 1.0
            
        # Compute average pairwise dissimilarity
        similarities = []
        indices_list = list(selected_indices)
        
        for i in range(len(indices_list)):
            for j in range(i + 1, len(indices_list)):
                idx1, idx2 = indices_list[i], indices_list[j]
                features1 = self.file_features[idx1]
                features2 = self.file_features[idx2]
                similarity = self._compute_similarity(features1, features2)
                similarities.append(similarity)
                
        if not similarities:
            return 1.0
            
        avg_similarity = sum(similarities) / len(similarities)
        return 1.0 - avg_similarity  # Convert to diversity score


def create_facility_selector(diversity_weight: float = 0.3, 
                           coverage_weight: float = 0.7) -> FastFacilityLocation:
    """Create a facility location selector."""
    return FastFacilityLocation(diversity_weight, coverage_weight)
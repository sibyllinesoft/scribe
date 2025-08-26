"""
Maximal Marginal Relevance (MMR) with sparse features for FastPath selection.

Implements MMR algorithm optimized for sparse TF-IDF features:
- Balances relevance (quality score) vs redundancy (similarity)
- Uses sparse vector operations for efficiency  
- Includes demotion ladder for graceful token budget management
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..fastpath.fast_scan import ScanResult
from ..fastpath.heuristics import ScoreComponents


@dataclass
class MMRConfig:
    """Configuration for MMR selection."""
    lambda_param: float = 0.7      # Balance between relevance and diversity (0-1)
    diversity_threshold: float = 0.8  # Minimum diversity score to consider
    demotion_factor: float = 0.5    # Score reduction for demoted items
    max_iterations: int = 1000      # Maximum selection iterations


class MMRSelector:
    """
    Maximal Marginal Relevance selector with sparse features.
    
    Optimizes the MMR objective:
    MMR = λ * Relevance(d) - (1-λ) * max_similarity(d, selected)
    
    Uses sparse TF-IDF features for efficient similarity computation.
    """
    
    def __init__(self, config: Optional[MMRConfig] = None):
        self.config = config or MMRConfig()
        
        # Sparse feature vectors for files
        self.file_features: Dict[int, Dict[int, float]] = {}
        self.relevance_scores: Dict[int, float] = {}
        
    def _extract_sparse_features(self, result: ScanResult) -> Dict[int, float]:
        """Extract sparse feature vector from scan result."""
        features = {}
        feature_idx = 0
        
        # Path component features
        path_parts = result.stats.path.lower().replace('/', ' ').replace('\\', ' ').split()
        for part in path_parts:
            if len(part) >= 2:
                features[feature_idx] = 1.0
                feature_idx += 1
                
        # Language feature
        if result.stats.language:
            features[feature_idx] = 2.0  # Higher weight for language
            feature_idx += 1
            
        # File type features
        type_weight = 1.5
        if result.stats.is_readme:
            features[feature_idx] = type_weight * 2.0  # README very important
            feature_idx += 1
        if result.stats.is_test:
            features[feature_idx] = type_weight
            feature_idx += 1
        if result.stats.is_config:
            features[feature_idx] = type_weight
            feature_idx += 1
        if result.stats.is_docs:
            features[feature_idx] = type_weight
            feature_idx += 1
            
        # Import features (simplified)
        if result.imports and result.imports.imports:
            # Use import count as feature intensity
            import_weight = min(len(result.imports.imports) / 10.0, 2.0)
            features[feature_idx] = import_weight
            feature_idx += 1
            
            # Add features for external vs relative imports
            if result.imports.external_imports > 0:
                features[feature_idx] = result.imports.external_imports / 5.0
                feature_idx += 1
                
        # Document structure features
        if result.doc_analysis:
            doc = result.doc_analysis
            
            if doc.heading_count > 0:
                features[feature_idx] = min(doc.heading_count / 5.0, 2.0)
                feature_idx += 1
                
            if doc.toc_indicators > 0:
                features[feature_idx] = 2.0  # TOC is valuable
                feature_idx += 1
                
            if doc.link_count > 0:
                features[feature_idx] = min(doc.link_count / 10.0, 1.5)
                feature_idx += 1
                
        # Size-based features (normalized)
        size_feature = min(result.stats.size_bytes / 10000.0, 2.0)  # Normalize to reasonable range
        features[feature_idx] = size_feature
        feature_idx += 1
        
        # Depth feature (inverted - shallower files more important)
        depth_feature = 2.0 / max(result.stats.depth, 1)
        features[feature_idx] = depth_feature
        
        return features
        
    def _compute_cosine_similarity(self, features1: Dict[int, float], 
                                 features2: Dict[int, float]) -> float:
        """Compute cosine similarity between sparse feature vectors."""
        # Find common features
        common_features = set(features1.keys()) & set(features2.keys())
        
        if not common_features:
            return 0.0
            
        # Compute dot product
        dot_product = sum(features1[feat] * features2[feat] for feat in common_features)
        
        # Compute norms
        norm1_squared = sum(val * val for val in features1.values())
        norm2_squared = sum(val * val for val in features2.values())
        
        if norm1_squared == 0.0 or norm2_squared == 0.0:
            return 0.0
            
        # Return cosine similarity
        return dot_product / (math.sqrt(norm1_squared) * math.sqrt(norm2_squared))
        
    def _compute_max_similarity(self, candidate_idx: int, selected_indices: Set[int]) -> float:
        """Compute maximum similarity between candidate and selected files."""
        if not selected_indices:
            return 0.0
            
        candidate_features = self.file_features[candidate_idx]
        max_sim = 0.0
        
        for selected_idx in selected_indices:
            selected_features = self.file_features[selected_idx]
            similarity = self._compute_cosine_similarity(candidate_features, selected_features)
            max_sim = max(max_sim, similarity)
            
        return max_sim
        
    def _compute_mmr_score(self, candidate_idx: int, selected_indices: Set[int]) -> float:
        """Compute MMR score for candidate file."""
        relevance = self.relevance_scores[candidate_idx]
        max_similarity = self._compute_max_similarity(candidate_idx, selected_indices)
        
        # MMR formula
        mmr = (self.config.lambda_param * relevance - 
               (1 - self.config.lambda_param) * max_similarity)
        
        return mmr
        
    def _estimate_token_count(self, result: ScanResult) -> int:
        """Estimate token count for file."""
        # Simple estimation based on file size and type
        chars_per_token = 4  # Average characters per token
        
        # Adjust for file type
        if result.stats.is_docs or result.stats.language in {'markdown', 'text'}:
            chars_per_token = 5  # Text files have longer tokens on average
        elif result.stats.language in {'python', 'javascript', 'typescript'}:
            chars_per_token = 3.5  # Code has shorter tokens
            
        estimated_tokens = max(result.stats.size_bytes // chars_per_token, result.stats.lines)
        return int(estimated_tokens)
        
    def select_files(self, scored_files: List[Tuple[ScanResult, ScoreComponents]], 
                    token_budget: int) -> List[ScanResult]:
        """
        Select files using MMR algorithm within token budget.
        
        Returns ordered list of selected files optimizing MMR objective.
        """
        if not scored_files:
            return []
            
        # Extract features and relevance scores
        for idx, (result, score_components) in enumerate(scored_files):
            self.file_features[idx] = self._extract_sparse_features(result)
            self.relevance_scores[idx] = score_components.final_score
            
        # Normalize relevance scores to [0, 1]
        max_relevance = max(self.relevance_scores.values()) if self.relevance_scores else 1.0
        if max_relevance > 0:
            for idx in self.relevance_scores:
                self.relevance_scores[idx] /= max_relevance
                
        # MMR selection with demotion ladder
        selected_indices = set()
        selected_files = []
        total_tokens = 0
        demotion_candidates = set()  # Files to try with reduced priority
        
        # Always prioritize README files first
        readme_priority = []
        other_candidates = []
        
        for idx, (result, score_components) in enumerate(scored_files):
            if result.stats.is_readme:
                readme_priority.append(idx)
            else:
                other_candidates.append(idx)
                
        # Process README files first
        for idx in readme_priority:
            result, score_components = scored_files[idx]
            estimated_tokens = self._estimate_token_count(result)
            
            if total_tokens + estimated_tokens <= token_budget:
                selected_indices.add(idx)
                selected_files.append(result)
                total_tokens += estimated_tokens
                
        # Process remaining candidates with MMR
        candidates = [idx for idx in other_candidates if idx not in selected_indices]
        
        for iteration in range(self.config.max_iterations):
            if not candidates or total_tokens >= token_budget:
                break
                
            best_idx = None
            best_mmr = float('-inf')
            best_tokens = 0
            
            # Find candidate with highest MMR score that fits budget
            for idx in candidates:
                result, score_components = scored_files[idx]
                estimated_tokens = self._estimate_token_count(result)
                
                # Check if fits in remaining budget
                if total_tokens + estimated_tokens > token_budget:
                    # Add to demotion candidates for later consideration
                    demotion_candidates.add(idx)
                    continue
                    
                # Compute MMR score
                mmr_score = self._compute_mmr_score(idx, selected_indices)
                
                # Penalize if too similar to selected files
                max_sim = self._compute_max_similarity(idx, selected_indices)
                if max_sim > self.config.diversity_threshold:
                    mmr_score *= 0.5  # Reduce score for redundant files
                    
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx
                    best_tokens = estimated_tokens
                    
            # Select best candidate
            if best_idx is not None:
                selected_indices.add(best_idx)
                result, score_components = scored_files[best_idx]
                selected_files.append(result)
                total_tokens += best_tokens
                candidates.remove(best_idx)
            else:
                # Try demotion ladder - reduce quality scores and retry
                if demotion_candidates:
                    self._apply_demotion(demotion_candidates)
                    candidates.extend(list(demotion_candidates))
                    demotion_candidates.clear()
                else:
                    break
                    
        return selected_files
        
    def _apply_demotion(self, candidate_indices: Set[int]) -> None:
        """Apply demotion factor to candidate relevance scores."""
        for idx in candidate_indices:
            if idx in self.relevance_scores:
                self.relevance_scores[idx] *= self.config.demotion_factor
                
    def get_selection_statistics(self, selected_files: List[ScanResult]) -> Dict[str, float]:
        """Get statistics about the selection quality."""
        if not selected_files:
            return {}
            
        # Count file types
        readme_count = sum(1 for f in selected_files if f.stats.is_readme)
        test_count = sum(1 for f in selected_files if f.stats.is_test)  
        doc_count = sum(1 for f in selected_files if f.stats.is_docs)
        code_count = len(selected_files) - readme_count - test_count - doc_count
        
        # Language distribution
        languages = {}
        for file in selected_files:
            lang = file.stats.language or 'unknown'
            languages[lang] = languages.get(lang, 0) + 1
            
        # Average depth
        avg_depth = sum(f.stats.depth for f in selected_files) / len(selected_files)
        
        # Token estimate
        total_tokens = sum(self._estimate_token_count(f) for f in selected_files)
        
        return {
            'total_files': len(selected_files),
            'readme_files': readme_count,
            'test_files': test_count,
            'doc_files': doc_count,
            'code_files': code_count,
            'avg_depth': avg_depth,
            'estimated_tokens': total_tokens,
            'language_distribution': languages,
        }


def create_mmr_selector(lambda_param: float = 0.7, 
                       diversity_threshold: float = 0.8) -> MMRSelector:
    """Create an MMR selector with specified parameters."""
    config = MMRConfig(lambda_param=lambda_param, diversity_threshold=diversity_threshold)
    return MMRSelector(config)
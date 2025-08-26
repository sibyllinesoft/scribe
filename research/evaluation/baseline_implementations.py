#!/usr/bin/env python3
"""
Baseline System Implementations for FastPath Research
=====================================================

Complete implementations of all baseline systems for comparative evaluation:
- Naive TF-IDF: Simple keyword-based retrieval
- BM25: Traditional chunk-based retrieval with BM25 scoring  
- Random: Random file selection within budget
- FastPath V1: File-based selection with heuristic scoring
- FastPath V2: Enhanced with PageRank centrality
- FastPath V3: Complete with quota system and routing

All systems implement the same interface for fair comparison.
"""

import os
import re
import json
import math
import random
import hashlib
import networkx as nx
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from collections import defaultdict, Counter
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class FileContent:
    """Represents file content with metadata."""
    path: str
    content: str
    size: int
    language: str
    tokens: int
    
    def __post_init__(self):
        if self.tokens == 0:
            self.tokens = len(self.content.split())


class BaseRetriever(ABC):
    """Abstract base class for all retrieval systems."""
    
    def __init__(self, name: str):
        self.name = name
        self.token_budget = 100000
        self.configured = False
    
    def configure(self, token_budget: int = 100000, **kwargs) -> None:
        """Configure the retrieval system."""
        self.token_budget = token_budget
        self.configured = True
        self._additional_configuration(**kwargs)
    
    def _additional_configuration(self, **kwargs) -> None:
        """Additional configuration specific to each system."""
        pass
    
    @abstractmethod
    def retrieve(self, repository: 'Repository') -> Union[str, List[str]]:
        """Retrieve relevant content from repository within token budget."""
        pass
    
    def _count_tokens(self, text: str) -> int:
        """Simple token counting."""
        return len(text.split())
    
    def _extract_code_features(self, content: str, file_path: str) -> Dict[str, Any]:
        """Extract features from code content."""
        features = {
            'line_count': len(content.split('\n')),
            'char_count': len(content),
            'function_count': len(re.findall(r'\bdef\s+\w+|function\s+\w+|class\s+\w+', content, re.IGNORECASE)),
            'import_count': len(re.findall(r'^import\s+|^from\s+.*import', content, re.MULTILINE)),
            'comment_density': len(re.findall(r'#.*|//.*|/\*.*?\*/', content)) / max(len(content.split('\n')), 1),
            'file_extension': Path(file_path).suffix.lower(),
            'is_test': 'test' in file_path.lower() or 'spec' in file_path.lower(),
            'is_config': any(ext in Path(file_path).name.lower() for ext in ['.json', '.yaml', '.yml', '.toml', '.ini']),
            'is_documentation': any(ext in Path(file_path).suffix.lower() for ext in ['.md', '.rst', '.txt'])
        }
        return features


class NaiveTFIDFRetriever(BaseRetriever):
    """
    Naive TF-IDF based retrieval system.
    
    Uses simple keyword matching with TF-IDF scoring to select files.
    No sophisticated chunking or contextual understanding.
    """
    
    def __init__(self):
        super().__init__("Naive TF-IDF")
        self.vectorizer = None
        self.query_keywords = [
            "function", "class", "import", "def", "return", "if", "for", "while",
            "error", "exception", "bug", "fix", "feature", "implementation", "api",
            "test", "config", "setup", "main", "init", "util", "helper"
        ]
    
    def _additional_configuration(self, **kwargs):
        """Configure TF-IDF vectorizer."""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
    
    def retrieve(self, repository: 'Repository') -> List[str]:
        """Retrieve files using TF-IDF similarity to query keywords."""
        if not self.configured:
            raise RuntimeError("Retriever not configured. Call configure() first.")
        
        # Get all text files from repository
        files = repository.get_text_files()
        if not files:
            return []
        
        # Prepare documents
        documents = []
        file_paths = []
        
        for file_path in files:
            try:
                content = repository.read_file(file_path)
                if content and len(content.strip()) > 0:
                    documents.append(content)
                    file_paths.append(file_path)
            except Exception:
                continue
        
        if not documents:
            return []
        
        # Fit TF-IDF vectorizer
        try:
            tfidf_matrix = self.vectorizer.fit_transform(documents)
        except Exception:
            # Fallback to simple selection if TF-IDF fails
            return self._fallback_selection(files, repository)
        
        # Create query from keywords
        query = " ".join(self.query_keywords)
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Rank files by similarity
        ranked_indices = np.argsort(similarities)[::-1]
        
        # Select files within token budget
        selected_content = []
        total_tokens = 0
        
        for idx in ranked_indices:
            file_path = file_paths[idx]
            content = documents[idx]
            tokens = self._count_tokens(content)
            
            if total_tokens + tokens <= self.token_budget:
                selected_content.append(f"=== {file_path} ===\n{content}\n")
                total_tokens += tokens
            else:
                break
        
        return selected_content
    
    def _fallback_selection(self, files: List[str], repository: 'Repository') -> List[str]:
        """Fallback selection if TF-IDF fails."""
        selected_content = []
        total_tokens = 0
        
        # Prioritize important file types
        priority_extensions = ['.py', '.js', '.java', '.cpp', '.c', '.h', '.md']
        priority_names = ['main', 'index', 'app', 'server', 'client']
        
        sorted_files = sorted(files, key=lambda f: (
            -any(ext in f.lower() for ext in priority_extensions),
            -any(name in Path(f).stem.lower() for name in priority_names),
            len(f)  # Shorter paths first
        ))
        
        for file_path in sorted_files:
            try:
                content = repository.read_file(file_path)
                if content:
                    tokens = self._count_tokens(content)
                    if total_tokens + tokens <= self.token_budget:
                        selected_content.append(f"=== {file_path} ===\n{content}\n")
                        total_tokens += tokens
                    else:
                        break
            except Exception:
                continue
        
        return selected_content


class BM25Retriever(BaseRetriever):
    """
    BM25-based retrieval system with chunking.
    
    Implements proper BM25 scoring with document chunking for more
    sophisticated retrieval than naive TF-IDF.
    """
    
    def __init__(self):
        super().__init__("BM25")
        self.k1 = 1.2  # Term frequency normalization
        self.b = 0.75  # Length normalization
        self.chunk_size = 500  # Words per chunk
        self.chunk_overlap = 50  # Word overlap between chunks
    
    def _additional_configuration(self, **kwargs):
        """Configure BM25 parameters."""
        self.k1 = kwargs.get('k1', 1.2)
        self.b = kwargs.get('b', 0.75)
        self.chunk_size = kwargs.get('chunk_size', 500)
        self.chunk_overlap = kwargs.get('chunk_overlap', 50)
    
    def retrieve(self, repository: 'Repository') -> List[str]:
        """Retrieve content using BM25 scoring on chunks."""
        if not self.configured:
            raise RuntimeError("Retriever not configured. Call configure() first.")
        
        # Get all text files
        files = repository.get_text_files()
        if not files:
            return []
        
        # Create chunks from all files
        chunks = []
        chunk_metadata = []
        
        for file_path in files:
            try:
                content = repository.read_file(file_path)
                if content and len(content.strip()) > 0:
                    file_chunks = self._create_chunks(content, file_path)
                    chunks.extend([chunk['content'] for chunk in file_chunks])
                    chunk_metadata.extend(file_chunks)
            except Exception:
                continue
        
        if not chunks:
            return []
        
        # Build BM25 index
        corpus_stats = self._build_bm25_index(chunks)
        
        # Define query terms (common programming concepts)
        query_terms = [
            'function', 'class', 'method', 'variable', 'import', 'export',
            'config', 'setup', 'init', 'main', 'api', 'endpoint', 'handler',
            'test', 'spec', 'mock', 'error', 'exception', 'log', 'debug'
        ]
        
        # Score all chunks
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            score = self._calculate_bm25_score(
                query_terms, chunk, corpus_stats, i
            )
            chunk_scores.append((i, score))
        
        # Sort by score
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select chunks within token budget
        selected_content = []
        total_tokens = 0
        used_files = set()
        
        for chunk_idx, score in chunk_scores:
            chunk_data = chunk_metadata[chunk_idx]
            content = chunk_data['content']
            file_path = chunk_data['file_path']
            
            tokens = self._count_tokens(content)
            
            if total_tokens + tokens <= self.token_budget:
                # Add file header if first chunk from this file
                if file_path not in used_files:
                    header = f"\n=== {file_path} ===\n"
                    selected_content.append(header)
                    used_files.add(file_path)
                
                selected_content.append(f"{content}\n\n")
                total_tokens += tokens
            else:
                break
        
        return selected_content
    
    def _create_chunks(self, content: str, file_path: str) -> List[Dict]:
        """Create overlapping chunks from content."""
        words = content.split()
        chunks = []
        
        if len(words) <= self.chunk_size:
            # Content fits in one chunk
            return [{
                'content': content,
                'file_path': file_path,
                'chunk_id': 0,
                'start_word': 0,
                'end_word': len(words)
            }]
        
        # Create overlapping chunks
        start = 0
        chunk_id = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_content = ' '.join(chunk_words)
            
            chunks.append({
                'content': chunk_content,
                'file_path': file_path,
                'chunk_id': chunk_id,
                'start_word': start,
                'end_word': end
            })
            
            chunk_id += 1
            
            # Move start position with overlap
            if end == len(words):
                break
            start = end - self.chunk_overlap
        
        return chunks
    
    def _build_bm25_index(self, chunks: List[str]) -> Dict:
        """Build BM25 index statistics."""
        # Tokenize all chunks
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        
        # Calculate document frequencies
        df = defaultdict(int)  # Document frequency for each term
        for tokens in tokenized_chunks:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1
        
        # Calculate average document length
        total_length = sum(len(tokens) for tokens in tokenized_chunks)
        avg_length = total_length / len(tokenized_chunks) if tokenized_chunks else 0
        
        return {
            'tokenized_chunks': tokenized_chunks,
            'df': df,
            'avg_length': avg_length,
            'total_chunks': len(chunks)
        }
    
    def _calculate_bm25_score(
        self, 
        query_terms: List[str], 
        chunk: str, 
        corpus_stats: Dict, 
        chunk_idx: int
    ) -> float:
        """Calculate BM25 score for a chunk."""
        tokens = corpus_stats['tokenized_chunks'][chunk_idx]
        tf = Counter(tokens)
        chunk_length = len(tokens)
        avg_length = corpus_stats['avg_length']
        total_chunks = corpus_stats['total_chunks']
        
        score = 0.0
        for term in query_terms:
            term = term.lower()
            
            if term in tf:
                # Term frequency component
                term_freq = tf[term]
                tf_component = (term_freq * (self.k1 + 1)) / (
                    term_freq + self.k1 * (1 - self.b + self.b * (chunk_length / avg_length))
                )
                
                # Inverse document frequency component
                df_term = corpus_stats['df'][term]
                idf_component = math.log((total_chunks - df_term + 0.5) / (df_term + 0.5))
                
                score += tf_component * idf_component
        
        return score


class RandomRetriever(BaseRetriever):
    """
    Random baseline that selects files randomly within token budget.
    
    Provides a lower bound for comparison - any intelligent system
    should perform better than random selection.
    """
    
    def __init__(self):
        super().__init__("Random")
        self.random_seed = 42
    
    def _additional_configuration(self, **kwargs):
        """Configure random seed."""
        self.random_seed = kwargs.get('random_seed', 42)
        random.seed(self.random_seed)
    
    def retrieve(self, repository: 'Repository') -> List[str]:
        """Randomly select files within token budget."""
        if not self.configured:
            raise RuntimeError("Retriever not configured. Call configure() first.")
        
        # Get all text files
        files = repository.get_text_files()
        if not files:
            return []
        
        # Shuffle files randomly
        shuffled_files = files.copy()
        random.shuffle(shuffled_files)
        
        # Select files within token budget
        selected_content = []
        total_tokens = 0
        
        for file_path in shuffled_files:
            try:
                content = repository.read_file(file_path)
                if content and len(content.strip()) > 0:
                    tokens = self._count_tokens(content)
                    
                    if total_tokens + tokens <= self.token_budget:
                        selected_content.append(f"=== {file_path} ===\n{content}\n")
                        total_tokens += tokens
                    else:
                        # Try to fit partial content
                        remaining_budget = self.token_budget - total_tokens
                        if remaining_budget > 100:  # Only if significant space left
                            words = content.split()
                            if len(words) > remaining_budget:
                                partial_content = ' '.join(words[:remaining_budget - 10])
                                selected_content.append(f"=== {file_path} (partial) ===\n{partial_content}...\n")
                        break
            except Exception:
                continue
        
        return selected_content


class FastPathV1(BaseRetriever):
    """
    FastPath Version 1: File-based selection with heuristic scoring.
    
    Implements basic FastPath algorithm with:
    - File importance scoring based on name, size, type
    - Heuristic rules for selecting relevant files
    - Simple dependency analysis
    """
    
    def __init__(self):
        super().__init__("FastPath V1")
        self.importance_weights = {
            'name_importance': 0.3,
            'size_importance': 0.2,
            'type_importance': 0.25,
            'dependency_importance': 0.25
        }
    
    def retrieve(self, repository: 'Repository') -> List[str]:
        """Retrieve files using FastPath V1 heuristics."""
        if not self.configured:
            raise RuntimeError("Retriever not configured. Call configure() first.")
        
        # Get all text files
        files = repository.get_text_files()
        if not files:
            return []
        
        # Score all files
        file_scores = []
        for file_path in files:
            try:
                content = repository.read_file(file_path)
                if content and len(content.strip()) > 0:
                    score = self._calculate_file_score(file_path, content, repository)
                    file_scores.append((file_path, content, score))
            except Exception:
                continue
        
        if not file_scores:
            return []
        
        # Sort by score (descending)
        file_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Select files within token budget
        selected_content = []
        total_tokens = 0
        
        for file_path, content, score in file_scores:
            tokens = self._count_tokens(content)
            
            if total_tokens + tokens <= self.token_budget:
                selected_content.append(f"=== {file_path} (score: {score:.3f}) ===\n{content}\n")
                total_tokens += tokens
            else:
                break
        
        return selected_content
    
    def _calculate_file_score(self, file_path: str, content: str, repository: 'Repository') -> float:
        """Calculate importance score for a file."""
        # Name importance
        name_score = self._calculate_name_importance(file_path)
        
        # Size importance (moderate size preferred)
        size_score = self._calculate_size_importance(content)
        
        # Type importance
        type_score = self._calculate_type_importance(file_path)
        
        # Dependency importance
        dependency_score = self._calculate_dependency_importance(file_path, content, repository)
        
        # Weighted combination
        total_score = (
            self.importance_weights['name_importance'] * name_score +
            self.importance_weights['size_importance'] * size_score +
            self.importance_weights['type_importance'] * type_score +
            self.importance_weights['dependency_importance'] * dependency_score
        )
        
        return total_score
    
    def _calculate_name_importance(self, file_path: str) -> float:
        """Score based on file name patterns."""
        path = Path(file_path)
        filename = path.stem.lower()
        
        # High importance names
        high_importance = ['main', 'index', 'app', 'server', 'client', 'core', 'base', 'utils', 'common']
        if filename in high_importance:
            return 1.0
        
        # Medium importance patterns
        if any(pattern in filename for pattern in ['config', 'setup', 'init', 'manager', 'handler', 'service']):
            return 0.7
        
        # Low importance patterns
        if any(pattern in filename for pattern in ['test', 'spec', 'mock', 'temp', 'cache']):
            return 0.3
        
        # Very low importance
        if any(pattern in filename for pattern in ['backup', 'old', 'deprecated', 'legacy']):
            return 0.1
        
        return 0.5  # Default score
    
    def _calculate_size_importance(self, content: str) -> float:
        """Score based on content size (moderate size preferred)."""
        lines = len(content.split('\n'))
        
        if lines < 10:
            return 0.2  # Too small
        elif lines < 50:
            return 0.8  # Good size
        elif lines < 200:
            return 1.0  # Ideal size
        elif lines < 500:
            return 0.7  # Getting large
        elif lines < 1000:
            return 0.4  # Large file
        else:
            return 0.2  # Very large file
    
    def _calculate_type_importance(self, file_path: str) -> float:
        """Score based on file type."""
        ext = Path(file_path).suffix.lower()
        
        # Source code files
        source_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.go', '.rs', '.rb']
        if ext in source_extensions:
            return 1.0
        
        # Configuration files
        config_extensions = ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg']
        if ext in config_extensions:
            return 0.8
        
        # Documentation files
        doc_extensions = ['.md', '.rst', '.txt']
        if ext in doc_extensions:
            return 0.6
        
        # Data files
        data_extensions = ['.csv', '.xml', '.sql']
        if ext in data_extensions:
            return 0.4
        
        return 0.3  # Other files
    
    def _calculate_dependency_importance(self, file_path: str, content: str, repository: 'Repository') -> float:
        """Score based on how many other files depend on this one."""
        # Simple heuristic: count imports/includes of this file
        path_stem = Path(file_path).stem
        
        import_count = 0
        all_files = repository.get_text_files()
        
        for other_file_path in all_files[:50]:  # Limit for performance
            if other_file_path == file_path:
                continue
            
            try:
                other_content = repository.read_file(other_file_path)
                if other_content:
                    # Count references to this file
                    import_patterns = [
                        f'import.*{path_stem}',
                        f'from.*{path_stem}',
                        f'require.*{path_stem}',
                        f'#include.*{path_stem}'
                    ]
                    
                    for pattern in import_patterns:
                        if re.search(pattern, other_content, re.IGNORECASE):
                            import_count += 1
                            break
            except Exception:
                continue
        
        # Normalize to 0-1 range
        return min(import_count / 10.0, 1.0)


class FastPathV2(FastPathV1):
    """
    FastPath Version 2: Enhanced with PageRank centrality.
    
    Extends V1 with:
    - Dependency graph construction
    - PageRank centrality analysis
    - Graph-based importance scoring
    """
    
    def __init__(self):
        super().__init__()
        self.name = "FastPath V2"
        self.pagerank_weight = 0.4
        self.importance_weights = {
            'name_importance': 0.2,
            'size_importance': 0.15,
            'type_importance': 0.15,
            'dependency_importance': 0.1,
            'pagerank_importance': 0.4
        }
    
    def retrieve(self, repository: 'Repository') -> List[str]:
        """Retrieve files using FastPath V2 with PageRank."""
        if not self.configured:
            raise RuntimeError("Retriever not configured. Call configure() first.")
        
        # Get all text files
        files = repository.get_text_files()
        if not files:
            return []
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(files, repository)
        
        # Calculate PageRank scores
        pagerank_scores = self._calculate_pagerank_scores(dependency_graph)
        
        # Score all files (including PageRank)
        file_scores = []
        for file_path in files:
            try:
                content = repository.read_file(file_path)
                if content and len(content.strip()) > 0:
                    base_score = self._calculate_file_score_v1(file_path, content, repository)
                    pagerank_score = pagerank_scores.get(file_path, 0)
                    
                    # Combine scores
                    total_score = (
                        (1 - self.pagerank_weight) * base_score +
                        self.pagerank_weight * pagerank_score
                    )
                    
                    file_scores.append((file_path, content, total_score))
            except Exception:
                continue
        
        if not file_scores:
            return []
        
        # Sort by score (descending)
        file_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Select files within token budget
        selected_content = []
        total_tokens = 0
        
        for file_path, content, score in file_scores:
            tokens = self._count_tokens(content)
            
            if total_tokens + tokens <= self.token_budget:
                selected_content.append(f"=== {file_path} (score: {score:.3f}) ===\n{content}\n")
                total_tokens += tokens
            else:
                break
        
        return selected_content
    
    def _calculate_file_score_v1(self, file_path: str, content: str, repository: 'Repository') -> float:
        """Calculate V1 score without PageRank."""
        # Use parent class method but adjust weights
        name_score = self._calculate_name_importance(file_path)
        size_score = self._calculate_size_importance(content)
        type_score = self._calculate_type_importance(file_path)
        dependency_score = self._calculate_dependency_importance(file_path, content, repository)
        
        total_score = (
            self.importance_weights['name_importance'] * name_score +
            self.importance_weights['size_importance'] * size_score +
            self.importance_weights['type_importance'] * type_score +
            self.importance_weights['dependency_importance'] * dependency_score
        )
        
        return total_score
    
    def _build_dependency_graph(self, files: List[str], repository: 'Repository') -> nx.DiGraph:
        """Build directed dependency graph."""
        graph = nx.DiGraph()
        
        # Add all files as nodes
        for file_path in files:
            graph.add_node(file_path)
        
        # Add dependency edges
        for file_path in files:
            try:
                content = repository.read_file(file_path)
                if not content:
                    continue
                
                # Find dependencies
                dependencies = self._extract_dependencies(file_path, content, files)
                
                for dep in dependencies:
                    if dep in files:
                        graph.add_edge(file_path, dep)  # file_path depends on dep
                        
            except Exception:
                continue
        
        return graph
    
    def _extract_dependencies(self, file_path: str, content: str, all_files: List[str]) -> List[str]:
        """Extract file dependencies from content."""
        dependencies = []
        
        # Create mapping of file stems to full paths
        stem_to_path = {}
        for f in all_files:
            stem = Path(f).stem
            if stem not in stem_to_path:
                stem_to_path[stem] = []
            stem_to_path[stem].append(f)
        
        # Common import/include patterns
        patterns = [
            r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # Python imports
            r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import',  # Python from imports
            r'require\s*\(\s*["\']([^"\']+)["\']\s*\)',  # JavaScript require
            r'import\s+.*from\s+["\']([^"\']+)["\']',  # ES6 imports
            r'#include\s*[<"]([^>"]+)[>"]',  # C/C++ includes
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                # Look for matching files
                if match in stem_to_path:
                    dependencies.extend(stem_to_path[match])
                else:
                    # Try partial matching
                    for stem, paths in stem_to_path.items():
                        if match in stem or stem in match:
                            dependencies.extend(paths)
        
        return list(set(dependencies))
    
    def _calculate_pagerank_scores(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate PageRank scores for the dependency graph."""
        try:
            if len(graph.nodes()) == 0:
                return {}
            
            # Calculate PageRank
            pagerank = nx.pagerank(graph, alpha=0.85, max_iter=100, tol=1e-6)
            
            # Normalize scores to 0-1 range
            max_score = max(pagerank.values()) if pagerank else 1.0
            normalized_scores = {
                node: score / max_score for node, score in pagerank.items()
            } if max_score > 0 else {node: 0 for node in pagerank.keys()}
            
            return normalized_scores
            
        except Exception:
            # Fallback: uniform scores
            return {node: 0.5 for node in graph.nodes()}


class FastPathV3(FastPathV2):
    """
    FastPath Version 3: Complete system with quota and routing.
    
    Implements the full FastPath algorithm with:
    - Multi-level file scoring
    - Quota-based selection system
    - Intelligent routing between file types
    - Dynamic budget allocation
    """
    
    def __init__(self):
        super().__init__()
        self.name = "FastPath V3"
        self.quota_system = {
            'core_files': 0.4,      # Main implementation files
            'config_files': 0.15,   # Configuration and setup
            'test_files': 0.1,      # Test files
            'doc_files': 0.15,      # Documentation
            'utility_files': 0.2    # Utilities and helpers
        }
    
    def retrieve(self, repository: 'Repository') -> List[str]:
        """Retrieve files using FastPath V3 with quota system."""
        if not self.configured:
            raise RuntimeError("Retriever not configured. Call configure() first.")
        
        # Get all text files
        files = repository.get_text_files()
        if not files:
            return []
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(files, repository)
        pagerank_scores = self._calculate_pagerank_scores(dependency_graph)
        
        # Classify and score files
        classified_files = self._classify_and_score_files(files, repository, pagerank_scores)
        
        # Apply quota-based selection
        selected_content = self._apply_quota_selection(classified_files, repository)
        
        return selected_content
    
    def _classify_and_score_files(
        self, 
        files: List[str], 
        repository: 'Repository', 
        pagerank_scores: Dict[str, float]
    ) -> Dict[str, List[Tuple[str, str, float]]]:
        """Classify files into categories and score them."""
        classified = {
            'core_files': [],
            'config_files': [],
            'test_files': [],
            'doc_files': [],
            'utility_files': []
        }
        
        for file_path in files:
            try:
                content = repository.read_file(file_path)
                if not content or len(content.strip()) == 0:
                    continue
                
                # Classify file
                category = self._classify_file(file_path, content)
                
                # Calculate comprehensive score
                score = self._calculate_comprehensive_score(
                    file_path, content, repository, pagerank_scores
                )
                
                classified[category].append((file_path, content, score))
                
            except Exception:
                continue
        
        # Sort each category by score
        for category in classified:
            classified[category].sort(key=lambda x: x[2], reverse=True)
        
        return classified
    
    def _classify_file(self, file_path: str, content: str) -> str:
        """Classify file into one of the quota categories."""
        path = Path(file_path)
        filename = path.stem.lower()
        extension = path.suffix.lower()
        
        # Test files
        if any(pattern in filename for pattern in ['test', 'spec', 'mock']) or 'test' in str(path).lower():
            return 'test_files'
        
        # Documentation files
        if extension in ['.md', '.rst', '.txt'] or 'readme' in filename:
            return 'doc_files'
        
        # Configuration files
        if extension in ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf']:
            return 'config_files'
        
        if any(pattern in filename for pattern in ['config', 'setup', 'settings', 'env']):
            return 'config_files'
        
        # Core files (main implementation)
        if any(pattern in filename for pattern in ['main', 'index', 'app', 'server', 'client', 'core']):
            return 'core_files'
        
        if extension in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.go', '.rs']:
            # Analyze content to determine if core or utility
            features = self._extract_code_features(content, file_path)
            
            if features['function_count'] > 5 or features['line_count'] > 100:
                return 'core_files'
            else:
                return 'utility_files'
        
        return 'utility_files'
    
    def _calculate_comprehensive_score(
        self,
        file_path: str,
        content: str,
        repository: 'Repository',
        pagerank_scores: Dict[str, float]
    ) -> float:
        """Calculate comprehensive importance score."""
        # Base V2 score
        base_score = self._calculate_file_score_v1(file_path, content, repository)
        pagerank_score = pagerank_scores.get(file_path, 0)
        
        # Additional V3 factors
        complexity_score = self._calculate_complexity_score(content)
        recency_score = self._calculate_recency_score(file_path, repository)
        connectivity_score = self._calculate_connectivity_score(file_path, repository)
        
        # Weighted combination
        total_score = (
            0.3 * base_score +
            0.25 * pagerank_score +
            0.2 * complexity_score +
            0.15 * recency_score +
            0.1 * connectivity_score
        )
        
        return total_score
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Score based on code complexity."""
        features = self._extract_code_features(content, "")
        
        # Complexity indicators
        complexity_factors = [
            min(features['function_count'] / 10.0, 1.0),
            min(features['line_count'] / 200.0, 1.0),
            1.0 - features['comment_density'],  # Less comments = more complex
            min(features['import_count'] / 5.0, 1.0)
        ]
        
        return sum(complexity_factors) / len(complexity_factors)
    
    def _calculate_recency_score(self, file_path: str, repository: 'Repository') -> float:
        """Score based on file recency (mock implementation)."""
        # In real implementation, this would check git history
        # For now, prefer shorter paths (likely more recent/important)
        path_depth = len(Path(file_path).parts)
        return max(0, 1.0 - (path_depth - 1) * 0.2)
    
    def _calculate_connectivity_score(self, file_path: str, repository: 'Repository') -> float:
        """Score based on how well-connected the file is."""
        # Count references to this file in other files (already implemented in V1)
        return min(self._calculate_dependency_importance(file_path, "", repository) * 2, 1.0)
    
    def _apply_quota_selection(
        self, 
        classified_files: Dict[str, List[Tuple[str, str, float]]],
        repository: 'Repository'
    ) -> List[str]:
        """Apply quota-based selection with dynamic reallocation."""
        selected_content = []
        total_tokens = 0
        category_tokens = {category: 0 for category in self.quota_system.keys()}
        
        # Calculate token budgets per category
        category_budgets = {
            category: int(self.token_budget * quota)
            for category, quota in self.quota_system.items()
        }
        
        # First pass: select within quotas
        for category, budget in category_budgets.items():
            files = classified_files.get(category, [])
            
            for file_path, content, score in files:
                tokens = self._count_tokens(content)
                
                if category_tokens[category] + tokens <= budget:
                    selected_content.append(
                        f"=== {file_path} [{category}] (score: {score:.3f}) ===\n{content}\n"
                    )
                    category_tokens[category] += tokens
                    total_tokens += tokens
        
        # Second pass: reallocate unused quota
        remaining_budget = self.token_budget - total_tokens
        if remaining_budget > 0:
            # Collect remaining files
            remaining_files = []
            used_files = set()
            
            for content_block in selected_content:
                # Extract file path from content block
                lines = content_block.split('\n')
                if lines and lines[0].startswith('=== '):
                    file_path = lines[0].split(' [')[0].replace('=== ', '')
                    used_files.add(file_path)
            
            # Add unused files from all categories
            for category, files in classified_files.items():
                for file_path, content, score in files:
                    if file_path not in used_files:
                        remaining_files.append((file_path, content, score, category))
            
            # Sort by score and add best remaining files
            remaining_files.sort(key=lambda x: x[2], reverse=True)
            
            for file_path, content, score, category in remaining_files:
                tokens = self._count_tokens(content)
                if total_tokens + tokens <= self.token_budget:
                    selected_content.append(
                        f"=== {file_path} [{category}] (score: {score:.3f}) ===\n{content}\n"
                    )
                    total_tokens += tokens
                else:
                    break
        
        return selected_content


# Export all retriever classes
__all__ = [
    'BaseRetriever',
    'NaiveTFIDFRetriever', 
    'BM25Retriever',
    'RandomRetriever',
    'FastPathV1',
    'FastPathV2', 
    'FastPathV3'
]
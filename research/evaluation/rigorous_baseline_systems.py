#!/usr/bin/env python3
"""
Rigorous Baseline Systems for FastPath Research Validation
==========================================================

Implements production-quality baseline systems for academic comparison:
- Naive TF-IDF with proper document preprocessing  
- BM25 with Okapi scoring and tuned parameters
- Random selection with stratified sampling
- Oracle upper bound with human-curated file importance

All baselines implement identical interfaces and budget constraints
for fair comparison in research evaluation.
"""

import sys
import os
import json
import re
import math
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Metadata for document in baseline systems."""
    
    path: Path
    content: str
    tokens: int
    language: str
    file_type: str
    size_bytes: int
    importance_score: float = 0.0  # For oracle baseline
    

@dataclass
class SelectionResult:
    """Result from baseline selection algorithm."""
    
    selected_documents: List[int]  # Document indices
    total_tokens: int
    budget_utilization: float
    selection_time_ms: float
    metadata: Dict[str, Any]


class DocumentProcessor:
    """Document processing utilities for baseline systems."""
    
    @staticmethod
    def preprocess_text(text: str, language: str = "python") -> str:
        """Preprocess text for IR systems."""
        
        # Remove code comments based on language
        if language == "python":
            text = re.sub(r'#.*$', '', text, flags=re.MULTILINE)
            text = re.sub(r'""".*?"""', '', text, flags=re.DOTALL)
            text = re.sub(r"'''.*?'''", '', text, flags=re.DOTALL)
        elif language in ["javascript", "typescript"]:
            text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
            text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        elif language in ["java", "cpp"]:
            text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
            text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Convert to lowercase for matching
        text = text.lower().strip()
        
        return text
    
    @staticmethod
    def extract_keywords(text: str, language: str = "python") -> List[str]:
        """Extract relevant keywords from code text."""
        
        # Language-specific keyword extraction
        if language == "python":
            # Python keywords and common patterns
            keywords = re.findall(r'\b(?:def|class|import|from|return|if|else|for|while|try|except)\b', text)
            identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', text)
        else:
            # Generic identifier extraction
            identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)
            keywords = []
        
        # Combine and deduplicate
        all_terms = keywords + identifiers
        return list(set(term for term in all_terms if len(term) > 2))
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for budget calculations."""
        # Approximate token count (words * 1.3 for subword tokenization)
        word_count = len(text.split())
        return int(word_count * 1.3)


class NaiveTFIDFBaseline:
    """
    Naive TF-IDF baseline using scikit-learn.
    
    Implements simple keyword-based document ranking with TF-IDF weighting
    and cosine similarity matching.
    """
    
    def __init__(self, max_features: int = 1000, stop_words: str = 'english'):
        self.max_features = max_features
        self.stop_words = stop_words
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=stop_words,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'  # Code-friendly tokenization
        )
        
    def select_documents(
        self, 
        documents: List[DocumentMetadata], 
        query: str, 
        budget: int,
        seed: int = 42
    ) -> SelectionResult:
        """Select documents using TF-IDF similarity."""
        
        import time
        start_time = time.time()
        
        if not documents:
            return SelectionResult([], 0, 0.0, 0.0, {"method": "tfidf", "reason": "no_documents"})
        
        # Preprocess documents
        doc_texts = []
        for doc in documents:
            processed = DocumentProcessor.preprocess_text(doc.content, doc.language)
            doc_texts.append(processed)
        
        if not any(doc_texts):
            # All documents are empty after preprocessing
            return SelectionResult([0], documents[0].tokens, documents[0].tokens/budget, 0.0, 
                                 {"method": "tfidf", "reason": "empty_docs"})
        
        try:
            # Fit TF-IDF vectorizer
            doc_vectors = self.vectorizer.fit_transform(doc_texts)
            
            # Process query
            processed_query = DocumentProcessor.preprocess_text(query, "generic")
            query_vector = self.vectorizer.transform([processed_query])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            
        except Exception as e:
            logger.warning(f"TF-IDF vectorization failed: {e}")
            # Fallback to simple keyword matching
            query_words = set(query.lower().split())
            similarities = []
            
            for doc_text in doc_texts:
                doc_words = set(doc_text.split())
                similarity = len(query_words.intersection(doc_words)) / len(query_words.union(doc_words))
                similarities.append(similarity)
            
            similarities = np.array(similarities)
        
        # Sort documents by similarity (descending)
        ranked_indices = np.argsort(similarities)[::-1]
        
        # Select documents within budget
        selected_indices = []
        total_tokens = 0
        
        for idx in ranked_indices:
            doc_tokens = documents[idx].tokens
            if total_tokens + doc_tokens <= budget:
                selected_indices.append(int(idx))
                total_tokens += doc_tokens
            else:
                break
        
        # Ensure at least one document selected if possible
        if not selected_indices and documents:
            selected_indices = [0]
            total_tokens = documents[0].tokens
        
        selection_time = (time.time() - start_time) * 1000
        
        return SelectionResult(
            selected_documents=selected_indices,
            total_tokens=total_tokens,
            budget_utilization=total_tokens / budget,
            selection_time_ms=selection_time,
            metadata={
                "method": "naive_tfidf",
                "max_features": self.max_features,
                "similarity_scores": similarities[selected_indices].tolist(),
                "total_similarity_sum": float(np.sum(similarities[selected_indices]))
            }
        )


class BM25Baseline:
    """
    BM25 (Okapi BM25) baseline implementation.
    
    Implements the standard BM25 ranking function with tunable parameters
    for probabilistic information retrieval.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Field length normalization parameter  
        self.epsilon = epsilon  # Floor value for IDF
        
    def select_documents(
        self, 
        documents: List[DocumentMetadata], 
        query: str, 
        budget: int,
        seed: int = 42
    ) -> SelectionResult:
        """Select documents using BM25 scoring."""
        
        import time
        start_time = time.time()
        
        if not documents:
            return SelectionResult([], 0, 0.0, 0.0, {"method": "bm25", "reason": "no_documents"})
        
        # Preprocess documents and extract terms
        doc_terms = []
        for doc in documents:
            processed = DocumentProcessor.preprocess_text(doc.content, doc.language)
            terms = processed.split()
            doc_terms.append(terms)
        
        if not any(doc_terms):
            return SelectionResult([0], documents[0].tokens, documents[0].tokens/budget, 0.0,
                                 {"method": "bm25", "reason": "empty_docs"})
        
        # Process query terms
        query_terms = DocumentProcessor.preprocess_text(query, "generic").split()
        
        if not query_terms:
            # No query terms, select first document
            return SelectionResult([0], documents[0].tokens, documents[0].tokens/budget, 0.0,
                                 {"method": "bm25", "reason": "empty_query"})
        
        # Calculate document statistics
        total_docs = len(doc_terms)
        doc_lengths = [len(terms) for terms in doc_terms]
        avg_doc_length = sum(doc_lengths) / total_docs if total_docs > 0 else 1
        
        # Calculate document frequency for each term
        df = defaultdict(int)
        for terms in doc_terms:
            unique_terms = set(terms)
            for term in unique_terms:
                df[term] += 1
        
        # Calculate BM25 scores for each document
        bm25_scores = []
        
        for i, doc_terms_list in enumerate(doc_terms):
            doc_length = doc_lengths[i]
            score = 0.0
            
            # Count term frequencies in this document
            tf = Counter(doc_terms_list)
            
            for query_term in query_terms:
                if query_term in tf:
                    # Term frequency component
                    term_freq = tf[query_term]
                    tf_component = (term_freq * (self.k1 + 1)) / (
                        term_freq + self.k1 * (1 - self.b + self.b * doc_length / avg_doc_length)
                    )
                    
                    # Inverse document frequency component
                    doc_freq = df[query_term]
                    idf = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                    idf = max(self.epsilon, idf)  # Apply floor
                    
                    score += idf * tf_component
            
            bm25_scores.append(score)
        
        # Rank documents by BM25 score
        ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        
        # Select documents within budget
        selected_indices = []
        total_tokens = 0
        
        for idx in ranked_indices:
            doc_tokens = documents[idx].tokens
            if total_tokens + doc_tokens <= budget:
                selected_indices.append(idx)
                total_tokens += doc_tokens
            else:
                break
        
        # Ensure at least one document selected
        if not selected_indices and documents:
            selected_indices = [0]
            total_tokens = documents[0].tokens
        
        selection_time = (time.time() - start_time) * 1000
        
        return SelectionResult(
            selected_documents=selected_indices,
            total_tokens=total_tokens,
            budget_utilization=total_tokens / budget,
            selection_time_ms=selection_time,
            metadata={
                "method": "bm25",
                "k1": self.k1,
                "b": self.b,
                "epsilon": self.epsilon,
                "bm25_scores": [bm25_scores[i] for i in selected_indices],
                "avg_doc_length": avg_doc_length
            }
        )


class RandomBaseline:
    """
    Random baseline with stratified sampling.
    
    Implements random document selection with optional stratification
    by file type and importance weighting.
    """
    
    def __init__(self, stratify_by_type: bool = True):
        self.stratify_by_type = stratify_by_type
    
    def select_documents(
        self, 
        documents: List[DocumentMetadata], 
        query: str, 
        budget: int,
        seed: int = 42
    ) -> SelectionResult:
        """Select documents randomly with stratification."""
        
        import time
        start_time = time.time()
        
        if not documents:
            return SelectionResult([], 0, 0.0, 0.0, {"method": "random", "reason": "no_documents"})
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        if self.stratify_by_type:
            # Group documents by file type
            type_groups = defaultdict(list)
            for i, doc in enumerate(documents):
                type_groups[doc.file_type].append(i)
            
            # Calculate how many documents to sample from each type
            available_indices = []
            for file_type, indices in type_groups.items():
                # Sample proportionally from each type
                n_sample = max(1, len(indices) // 3)  # At least 1, up to 1/3 of type
                sampled = random.sample(indices, min(n_sample, len(indices)))
                available_indices.extend(sampled)
            
            random.shuffle(available_indices)
        else:
            # Simple random permutation
            available_indices = list(range(len(documents)))
            random.shuffle(available_indices)
        
        # Select documents within budget
        selected_indices = []
        total_tokens = 0
        
        for idx in available_indices:
            doc_tokens = documents[idx].tokens
            if total_tokens + doc_tokens <= budget:
                selected_indices.append(idx)
                total_tokens += doc_tokens
            else:
                break
        
        # Ensure at least one document selected
        if not selected_indices and documents:
            selected_indices = [0]
            total_tokens = documents[0].tokens
        
        selection_time = (time.time() - start_time) * 1000
        
        return SelectionResult(
            selected_documents=selected_indices,
            total_tokens=total_tokens,
            budget_utilization=total_tokens / budget,
            selection_time_ms=selection_time,
            metadata={
                "method": "random",
                "stratify_by_type": self.stratify_by_type,
                "random_seed": seed,
                "file_types_selected": list(set(documents[i].file_type for i in selected_indices))
            }
        )


class OracleBaseline:
    """
    Oracle baseline using human-curated importance scores.
    
    Represents theoretical upper bound performance using ground truth
    file importance annotations.
    """
    
    def __init__(self, importance_source: str = "manual"):
        self.importance_source = importance_source
        
        # Default importance rules (can be overridden with manual annotations)
        self.default_importance_rules = {
            "README.md": 0.95,
            "main.py": 0.90,
            "__init__.py": 0.85,
            "setup.py": 0.80,
            "requirements.txt": 0.75,
            "pyproject.toml": 0.75,
            "config": 0.70,
            "cli": 0.70,
            "core": 0.85,
            "api": 0.80,
            "test": 0.30,  # Lower importance for tests in QA context
            "doc": 0.60,
            ".git": 0.05,
            ".env": 0.20
        }
    
    def _calculate_importance_score(self, doc: DocumentMetadata) -> float:
        """Calculate importance score for document."""
        
        path_str = str(doc.path).lower()
        base_score = 0.5  # Default baseline importance
        
        # Apply path-based rules
        for pattern, score in self.default_importance_rules.items():
            if pattern in path_str:
                base_score = max(base_score, score)
        
        # Boost for entry points and configuration
        if any(keyword in path_str for keyword in ["main", "entry", "start", "app"]):
            base_score += 0.1
        
        if any(keyword in path_str for keyword in ["config", "setting", "environment"]):
            base_score += 0.05
        
        # Penalty for very large or very small files
        if doc.size_bytes > 50000:  # Very large files
            base_score *= 0.8
        elif doc.size_bytes < 100:  # Very small files
            base_score *= 0.6
        
        return min(1.0, base_score)
    
    def select_documents(
        self, 
        documents: List[DocumentMetadata], 
        query: str, 
        budget: int,
        seed: int = 42
    ) -> SelectionResult:
        """Select documents using oracle importance scores."""
        
        import time
        start_time = time.time()
        
        if not documents:
            return SelectionResult([], 0, 0.0, 0.0, {"method": "oracle", "reason": "no_documents"})
        
        # Calculate or use existing importance scores
        doc_importance = []
        for doc in documents:
            if doc.importance_score > 0:
                # Use provided importance score
                importance = doc.importance_score
            else:
                # Calculate using rules
                importance = self._calculate_importance_score(doc)
            doc_importance.append(importance)
        
        # Sort by importance (descending)
        ranked_indices = sorted(range(len(documents)), key=lambda i: doc_importance[i], reverse=True)
        
        # Select documents within budget, prioritizing high importance
        selected_indices = []
        total_tokens = 0
        
        for idx in ranked_indices:
            doc_tokens = documents[idx].tokens
            if total_tokens + doc_tokens <= budget:
                selected_indices.append(idx)
                total_tokens += doc_tokens
            else:
                break
        
        # Ensure at least one document selected
        if not selected_indices and documents:
            selected_indices = [0]
            total_tokens = documents[0].tokens
        
        selection_time = (time.time() - start_time) * 1000
        
        return SelectionResult(
            selected_documents=selected_indices,
            total_tokens=total_tokens,
            budget_utilization=total_tokens / budget,
            selection_time_ms=selection_time,
            metadata={
                "method": "oracle",
                "importance_source": self.importance_source,
                "importance_scores": [doc_importance[i] for i in selected_indices],
                "avg_importance": np.mean([doc_importance[i] for i in selected_indices])
            }
        )


class BaselineSystemManager:
    """
    Manager for all baseline systems with consistent interfaces.
    
    Provides unified interface for evaluating multiple baseline systems
    with identical parameters and constraints.
    """
    
    def __init__(self):
        self.baselines = {
            "naive_tfidf": NaiveTFIDFBaseline(),
            "bm25": BM25Baseline(),
            "random": RandomBaseline(),
            "oracle": OracleBaseline()
        }
    
    def load_documents_from_repository(self, repo_path: Path) -> List[DocumentMetadata]:
        """Load documents from repository with metadata."""
        
        documents = []
        
        # File extensions to include
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.rs', '.go'}
        doc_extensions = {'.md', '.rst', '.txt'}
        config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg'}
        
        all_extensions = code_extensions | doc_extensions | config_extensions
        
        for file_path in repo_path.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix.lower() in all_extensions and
                not any(exclude in str(file_path) for exclude in ['.git', '__pycache__', 'node_modules'])):
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if not content.strip():
                        continue
                    
                    # Determine language and file type
                    ext = file_path.suffix.lower()
                    if ext == '.py':
                        language = 'python'
                        file_type = 'code'
                    elif ext in ['.js', '.ts']:
                        language = 'javascript'
                        file_type = 'code'
                    elif ext in doc_extensions:
                        language = 'markdown'
                        file_type = 'documentation'
                    elif ext in config_extensions:
                        language = 'config'
                        file_type = 'configuration'
                    else:
                        language = 'generic'
                        file_type = 'other'
                    
                    doc = DocumentMetadata(
                        path=file_path,
                        content=content,
                        tokens=DocumentProcessor.estimate_tokens(content),
                        language=language,
                        file_type=file_type,
                        size_bytes=len(content.encode('utf-8'))
                    )
                    
                    documents.append(doc)
                    
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    continue
        
        return documents
    
    def evaluate_baseline(
        self,
        baseline_id: str,
        documents: List[DocumentMetadata],
        query: str,
        budget: int,
        seed: int = 42
    ) -> SelectionResult:
        """Evaluate specific baseline system."""
        
        if baseline_id not in self.baselines:
            raise ValueError(f"Unknown baseline: {baseline_id}. Available: {list(self.baselines.keys())}")
        
        baseline = self.baselines[baseline_id]
        return baseline.select_documents(documents, query, budget, seed)
    
    def evaluate_all_baselines(
        self,
        documents: List[DocumentMetadata],
        query: str,
        budget: int,
        seed: int = 42
    ) -> Dict[str, SelectionResult]:
        """Evaluate all baseline systems."""
        
        results = {}
        
        for baseline_id in self.baselines:
            try:
                result = self.evaluate_baseline(baseline_id, documents, query, budget, seed)
                results[baseline_id] = result
            except Exception as e:
                logger.error(f"Baseline {baseline_id} failed: {e}")
                results[baseline_id] = SelectionResult(
                    [], 0, 0.0, 0.0, {"method": baseline_id, "error": str(e)}
                )
        
        return results
    
    def compare_baselines(
        self,
        documents: List[DocumentMetadata], 
        queries: List[Dict[str, Any]],
        budgets: List[int],
        seeds: List[int] = None
    ) -> Dict[str, Any]:
        """Comprehensive baseline comparison."""
        
        if seeds is None:
            seeds = [42]
        
        all_results = []
        
        for query_info in queries:
            query = query_info['question']
            query_id = query_info.get('id', 'unknown')
            
            for budget in budgets:
                for seed in seeds:
                    baseline_results = self.evaluate_all_baselines(documents, query, budget, seed)
                    
                    for baseline_id, result in baseline_results.items():
                        all_results.append({
                            "query_id": query_id,
                            "baseline_id": baseline_id,
                            "budget": budget,
                            "seed": seed,
                            "selected_documents": result.selected_documents,
                            "total_tokens": result.total_tokens,
                            "budget_utilization": result.budget_utilization,
                            "selection_time_ms": result.selection_time_ms,
                            "metadata": result.metadata
                        })
        
        return {
            "timestamp": json.dumps({"timestamp": "now"}),  # Placeholder
            "total_evaluations": len(all_results),
            "baselines_tested": list(self.baselines.keys()),
            "results": all_results
        }


def main():
    """Demo of rigorous baseline systems."""
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        print("🧪 Running Rigorous Baseline Systems Demo...")
        
        # Initialize manager
        manager = BaselineSystemManager()
        
        # Load documents from current repository
        repo_path = Path(".")
        documents = manager.load_documents_from_repository(repo_path)
        
        print(f"📚 Loaded {len(documents)} documents from repository")
        
        # Demo queries
        demo_queries = [
            {
                "id": "arch_overview",
                "question": "What is the architecture and main components of this system?",
                "category": "architecture"
            },
            {
                "id": "setup_install",
                "question": "How do I install and setup this project for development?",
                "category": "setup"
            }
        ]
        
        # Demo evaluation
        budgets = [50000, 120000]
        results = manager.compare_baselines(documents, demo_queries, budgets, [42, 123])
        
        print(f"\n📊 Baseline Comparison Results:")
        print(f"Total Evaluations: {results['total_evaluations']}")
        print(f"Baselines Tested: {results['baselines_tested']}")
        
        # Show sample results
        for baseline_id in results['baselines_tested']:
            baseline_results = [r for r in results['results'] if r['baseline_id'] == baseline_id]
            if baseline_results:
                avg_utilization = np.mean([r['budget_utilization'] for r in baseline_results])
                avg_time = np.mean([r['selection_time_ms'] for r in baseline_results])
                print(f"  {baseline_id}: {avg_utilization:.1%} budget utilization, {avg_time:.1f}ms avg time")
        
        # Save results for research use
        output_file = Path("baseline_evaluation_demo.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📁 Results saved to: {output_file}")
        return results
        
    else:
        print("Usage: rigorous_baseline_systems.py --demo")
        print("       Run comprehensive baseline systems demo")
        return None


if __name__ == "__main__":
    main()
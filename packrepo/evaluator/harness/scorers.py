#!/usr/bin/env python3
"""
Scoring System - Answer Evaluation and Metrics

Implements comprehensive answer scoring according to TODO.md requirements:
- Exact match and regex pattern scoring for objective questions
- Semantic similarity scoring using embeddings
- Statistical validation and consistency checks
- Integration with judge system for subjective evaluation

Key Features:
- Multiple scoring methods (exact, regex, semantic, judge-based)
- Confidence-weighted scoring with uncertainty quantification
- Batch processing for efficiency
- Comprehensive metrics tracking and reporting
- Integration with evaluation pipeline
"""

import asyncio
import json
import logging
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import numpy as np
from collections import defaultdict

# Optional embedding imports
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None

from .judge import AnswerJudge, ExactMatchJudge, JudgmentDecision, create_judge
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class ScoreType(Enum):
    """Types of scoring methods."""
    EXACT_MATCH = "exact_match"
    REGEX_MATCH = "regex_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    BLIND_AB_JUDGE = "blind_ab_judge"
    COMPOSITE = "composite"


@dataclass
class ScoringCriteria:
    """Criteria for scoring an answer."""
    question_id: str
    score_type: ScoreType
    
    # For exact/regex matching
    expected_values: Optional[List[str]] = None
    regex_patterns: Optional[List[str]] = None
    case_sensitive: bool = False
    
    # For semantic similarity
    reference_answer: Optional[str] = None
    similarity_threshold: float = 0.7
    
    # For judge-based scoring
    comparison_answers: Optional[List[str]] = None
    judge_rubric: Optional[str] = None
    
    # Scoring weights
    weight: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ScoreResult:
    """Result of scoring an answer."""
    question_id: str
    answer_text: str
    score_type: ScoreType
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    explanation: str
    
    # Detailed breakdown
    component_scores: Dict[str, float] = None
    match_details: Dict[str, Any] = None
    
    # Metadata
    scorer_metadata: Dict[str, Any] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if self.component_scores is None:
            self.component_scores = {}
        if self.match_details is None:
            self.match_details = {}
        if self.scorer_metadata is None:
            self.scorer_metadata = {}
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class ExactMatchScorer:
    """Scorer using exact string matching."""
    
    def __init__(self):
        self.stats = defaultdict(int)
    
    def score(
        self, 
        answer: str, 
        criteria: ScoringCriteria
    ) -> ScoreResult:
        """Score answer using exact match criteria."""
        if not criteria.expected_values:
            return ScoreResult(
                question_id=criteria.question_id,
                answer_text=answer,
                score_type=ScoreType.EXACT_MATCH,
                score=0.0,
                confidence=1.0,
                explanation="No expected values provided for exact match"
            )
        
        answer_norm = answer if criteria.case_sensitive else answer.lower()
        matches = []
        match_details = {}
        
        for expected in criteria.expected_values:
            expected_norm = expected if criteria.case_sensitive else expected.lower()
            if expected_norm in answer_norm:
                matches.append(expected)
                match_details[expected] = {
                    "found": True,
                    "position": answer_norm.find(expected_norm)
                }
                self.stats["exact_matches"] += 1
            else:
                match_details[expected] = {"found": False}
                self.stats["exact_misses"] += 1
        
        # Calculate score based on match ratio
        score = len(matches) / len(criteria.expected_values)
        confidence = 1.0  # Exact matching has perfect confidence
        
        explanation = f"Found {len(matches)}/{len(criteria.expected_values)} exact matches: {matches}"
        
        return ScoreResult(
            question_id=criteria.question_id,
            answer_text=answer,
            score_type=ScoreType.EXACT_MATCH,
            score=score,
            confidence=confidence,
            explanation=explanation,
            match_details=match_details,
            scorer_metadata={"matches": matches, "total_expected": len(criteria.expected_values)}
        )


class RegexScorer:
    """Scorer using regular expression matching."""
    
    def __init__(self):
        self.stats = defaultdict(int)
        self.compiled_patterns = {}  # Cache compiled patterns
    
    def score(
        self, 
        answer: str, 
        criteria: ScoringCriteria
    ) -> ScoreResult:
        """Score answer using regex patterns."""
        if not criteria.regex_patterns:
            return ScoreResult(
                question_id=criteria.question_id,
                answer_text=answer,
                score_type=ScoreType.REGEX_MATCH,
                score=0.0,
                confidence=1.0,
                explanation="No regex patterns provided"
            )
        
        matches = []
        match_details = {}
        
        for pattern in criteria.regex_patterns:
            try:
                # Use cached compiled pattern if available
                if pattern not in self.compiled_patterns:
                    self.compiled_patterns[pattern] = re.compile(pattern, re.IGNORECASE)
                
                compiled_pattern = self.compiled_patterns[pattern]
                match = compiled_pattern.search(answer)
                
                if match:
                    matches.append(pattern)
                    match_details[pattern] = {
                        "found": True,
                        "match_text": match.group(),
                        "start": match.start(),
                        "end": match.end()
                    }
                    self.stats["regex_matches"] += 1
                else:
                    match_details[pattern] = {"found": False}
                    self.stats["regex_misses"] += 1
                    
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                match_details[pattern] = {"found": False, "error": str(e)}
                self.stats["regex_errors"] += 1
        
        # Calculate score based on match ratio
        valid_patterns = [p for p in criteria.regex_patterns if "error" not in match_details.get(p, {})]
        if not valid_patterns:
            score = 0.0
            confidence = 0.0  # No confidence if all patterns are invalid
        else:
            score = len(matches) / len(valid_patterns)
            confidence = 0.9  # Slightly lower confidence than exact match
        
        explanation = f"Found {len(matches)}/{len(valid_patterns)} regex matches"
        
        return ScoreResult(
            question_id=criteria.question_id,
            answer_text=answer,
            score_type=ScoreType.REGEX_MATCH,
            score=score,
            confidence=confidence,
            explanation=explanation,
            match_details=match_details,
            scorer_metadata={
                "matching_patterns": matches, 
                "total_patterns": len(criteria.regex_patterns),
                "valid_patterns": len(valid_patterns)
            }
        )


class SemanticSimilarityScorer:
    """Scorer using semantic similarity with embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
        
        self.model = SentenceTransformer(model_name)
        self.stats = defaultdict(int)
    
    def score(
        self, 
        answer: str, 
        criteria: ScoringCriteria
    ) -> ScoreResult:
        """Score answer using semantic similarity."""
        if not criteria.reference_answer:
            return ScoreResult(
                question_id=criteria.question_id,
                answer_text=answer,
                score_type=ScoreType.SEMANTIC_SIMILARITY,
                score=0.0,
                confidence=0.0,
                explanation="No reference answer provided for semantic similarity"
            )
        
        try:
            # Generate embeddings
            embeddings = self.model.encode([answer, criteria.reference_answer])
            answer_embedding, reference_embedding = embeddings[0], embeddings[1]
            
            # Calculate cosine similarity
            similarity = np.dot(answer_embedding, reference_embedding) / (
                np.linalg.norm(answer_embedding) * np.linalg.norm(reference_embedding)
            )
            
            # Convert to 0-1 scale
            similarity = max(0.0, float(similarity))
            
            # Apply threshold
            score = 1.0 if similarity >= criteria.similarity_threshold else similarity
            confidence = min(0.9, similarity + 0.1)  # Embedding similarity has inherent uncertainty
            
            if similarity >= criteria.similarity_threshold:
                self.stats["above_threshold"] += 1
                explanation = f"Semantic similarity {similarity:.3f} above threshold {criteria.similarity_threshold}"
            else:
                self.stats["below_threshold"] += 1
                explanation = f"Semantic similarity {similarity:.3f} below threshold {criteria.similarity_threshold}"
            
            return ScoreResult(
                question_id=criteria.question_id,
                answer_text=answer,
                score_type=ScoreType.SEMANTIC_SIMILARITY,
                score=score,
                confidence=confidence,
                explanation=explanation,
                match_details={
                    "similarity": similarity,
                    "threshold": criteria.similarity_threshold,
                    "above_threshold": similarity >= criteria.similarity_threshold
                },
                scorer_metadata={
                    "model": self.model.get_sentence_embedding_dimension(),
                    "embedding_model": type(self.model).__name__
                }
            )
            
        except Exception as e:
            logger.error(f"Semantic similarity scoring failed: {e}")
            self.stats["errors"] += 1
            return ScoreResult(
                question_id=criteria.question_id,
                answer_text=answer,
                score_type=ScoreType.SEMANTIC_SIMILARITY,
                score=0.0,
                confidence=0.0,
                explanation=f"Semantic scoring error: {str(e)}"
            )


class JudgeBasedScorer:
    """Scorer using blind A/B judge comparison."""
    
    def __init__(self, llm_client: LLMClient, judge_kwargs: Dict[str, Any] = None):
        self.llm_client = llm_client
        judge_kwargs = judge_kwargs or {}
        self.judge = AnswerJudge(llm_client, **judge_kwargs)
        self.stats = defaultdict(int)
    
    async def score_comparative(
        self,
        answer: str,
        criteria: ScoringCriteria,
        question: str
    ) -> ScoreResult:
        """Score answer using comparative judgment against reference answers."""
        if not criteria.comparison_answers:
            return ScoreResult(
                question_id=criteria.question_id,
                answer_text=answer,
                score_type=ScoreType.BLIND_AB_JUDGE,
                score=0.0,
                confidence=0.0,
                explanation="No comparison answers provided for judge-based scoring"
            )
        
        # Compare against each reference answer
        comparison_results = []
        total_wins = 0
        total_comparisons = len(criteria.comparison_answers)
        
        for i, reference_answer in enumerate(criteria.comparison_answers):
            try:
                comparison = await self.judge.compare_answers(
                    question=question,
                    answer_1=answer,
                    answer_2=reference_answer,
                    reference_1="candidate",
                    reference_2=f"reference_{i}",
                    context=None
                )
                
                comparison_results.append(comparison)
                
                # Count wins for the candidate answer
                if comparison.consensus_decision == JudgmentDecision.A_BETTER:
                    # Candidate was answer A and won
                    if comparison.answer_refs[0] == "candidate":
                        total_wins += 1
                elif comparison.consensus_decision == JudgmentDecision.B_BETTER:
                    # Candidate was answer B and won
                    if comparison.answer_refs[1] == "candidate":
                        total_wins += 1
                # Ties count as 0.5 wins
                elif comparison.consensus_decision == JudgmentDecision.TIE:
                    total_wins += 0.5
                
                self.stats["comparisons_completed"] += 1
                
            except Exception as e:
                logger.error(f"Judge comparison failed: {e}")
                self.stats["comparison_errors"] += 1
                continue
        
        # Calculate overall score
        if total_comparisons > 0:
            score = total_wins / total_comparisons
            
            # Calculate confidence from agreement rates
            agreement_rates = [comp.agreement_rate for comp in comparison_results]
            confidence = np.mean(agreement_rates) if agreement_rates else 0.0
        else:
            score = 0.0
            confidence = 0.0
        
        explanation = f"Won {total_wins}/{total_comparisons} judge comparisons (score: {score:.2f})"
        
        # Aggregate metadata
        total_cost = sum(comp.total_cost for comp in comparison_results)
        avg_latency = np.mean([comp.avg_latency for comp in comparison_results]) if comparison_results else 0
        
        return ScoreResult(
            question_id=criteria.question_id,
            answer_text=answer,
            score_type=ScoreType.BLIND_AB_JUDGE,
            score=score,
            confidence=confidence,
            explanation=explanation,
            match_details={
                "total_wins": total_wins,
                "total_comparisons": total_comparisons,
                "win_rate": score
            },
            scorer_metadata={
                "total_cost": total_cost,
                "avg_latency": avg_latency,
                "num_references": len(criteria.comparison_answers),
                "judge_stats": self.judge.get_reliability_metrics()
            }
        )


class CompositeScorer:
    """Composite scorer that combines multiple scoring methods."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None, embedding_model: str = "all-MiniLM-L6-v2"):
        self.exact_scorer = ExactMatchScorer()
        self.regex_scorer = RegexScorer()
        
        # Optional scorers that require additional dependencies
        self.semantic_scorer = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.semantic_scorer = SemanticSimilarityScorer(embedding_model)
            except Exception as e:
                logger.warning(f"Failed to initialize semantic scorer: {e}")
        
        self.judge_scorer = None
        if llm_client:
            self.judge_scorer = JudgeBasedScorer(llm_client)
        
        self.stats = defaultdict(int)
    
    async def score(
        self,
        answer: str,
        criteria: List[ScoringCriteria],
        question: str = ""
    ) -> Dict[str, ScoreResult]:
        """Score answer using multiple criteria."""
        results = {}
        
        for criterion in criteria:
            try:
                if criterion.score_type == ScoreType.EXACT_MATCH:
                    result = self.exact_scorer.score(answer, criterion)
                
                elif criterion.score_type == ScoreType.REGEX_MATCH:
                    result = self.regex_scorer.score(answer, criterion)
                
                elif criterion.score_type == ScoreType.SEMANTIC_SIMILARITY:
                    if self.semantic_scorer:
                        result = self.semantic_scorer.score(answer, criterion)
                    else:
                        result = ScoreResult(
                            question_id=criterion.question_id,
                            answer_text=answer,
                            score_type=ScoreType.SEMANTIC_SIMILARITY,
                            score=0.0,
                            confidence=0.0,
                            explanation="Semantic similarity scorer not available"
                        )
                
                elif criterion.score_type == ScoreType.BLIND_AB_JUDGE:
                    if self.judge_scorer:
                        result = await self.judge_scorer.score_comparative(answer, criterion, question)
                    else:
                        result = ScoreResult(
                            question_id=criterion.question_id,
                            answer_text=answer,
                            score_type=ScoreType.BLIND_AB_JUDGE,
                            score=0.0,
                            confidence=0.0,
                            explanation="Judge-based scorer not available (no LLM client)"
                        )
                
                else:
                    result = ScoreResult(
                        question_id=criterion.question_id,
                        answer_text=answer,
                        score_type=criterion.score_type,
                        score=0.0,
                        confidence=0.0,
                        explanation=f"Unknown score type: {criterion.score_type}"
                    )
                
                results[f"{criterion.score_type.value}_{criterion.question_id}"] = result
                self.stats[f"{criterion.score_type.value}_completed"] += 1
                
            except Exception as e:
                logger.error(f"Scoring failed for {criterion.score_type}: {e}")
                self.stats[f"{criterion.score_type.value}_errors"] += 1
                
                error_result = ScoreResult(
                    question_id=criterion.question_id,
                    answer_text=answer,
                    score_type=criterion.score_type,
                    score=0.0,
                    confidence=0.0,
                    explanation=f"Scoring error: {str(e)}"
                )
                results[f"{criterion.score_type.value}_{criterion.question_id}"] = error_result
        
        return results
    
    def calculate_composite_score(
        self,
        score_results: Dict[str, ScoreResult],
        weights: Optional[Dict[str, float]] = None
    ) -> ScoreResult:
        """Calculate weighted composite score from multiple results."""
        if not score_results:
            return ScoreResult(
                question_id="composite",
                answer_text="",
                score_type=ScoreType.COMPOSITE,
                score=0.0,
                confidence=0.0,
                explanation="No score results to composite"
            )
        
        # Default equal weights
        if weights is None:
            weights = {key: 1.0 for key in score_results.keys()}
        
        weighted_scores = []
        confidences = []
        explanations = []
        
        total_weight = 0.0
        for key, result in score_results.items():
            weight = weights.get(key, 1.0)
            weighted_scores.append(result.score * weight)
            confidences.append(result.confidence)
            explanations.append(f"{result.score_type.value}: {result.score:.2f}")
            total_weight += weight
        
        # Calculate composite metrics
        if total_weight > 0:
            composite_score = sum(weighted_scores) / total_weight
            composite_confidence = np.mean(confidences)
        else:
            composite_score = 0.0
            composite_confidence = 0.0
        
        explanation = f"Composite of {len(score_results)} scores: {', '.join(explanations)}"
        
        # Aggregate metadata
        component_scores = {key: result.score for key, result in score_results.items()}
        
        return ScoreResult(
            question_id="composite",
            answer_text=list(score_results.values())[0].answer_text if score_results else "",
            score_type=ScoreType.COMPOSITE,
            score=composite_score,
            confidence=composite_confidence,
            explanation=explanation,
            component_scores=component_scores,
            scorer_metadata={
                "weights": weights,
                "total_weight": total_weight,
                "num_components": len(score_results)
            }
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics from all scorers."""
        all_stats = {"composite": dict(self.stats)}
        
        if hasattr(self.exact_scorer, 'stats'):
            all_stats["exact_match"] = dict(self.exact_scorer.stats)
        
        if hasattr(self.regex_scorer, 'stats'):
            all_stats["regex"] = dict(self.regex_scorer.stats)
        
        if self.semantic_scorer and hasattr(self.semantic_scorer, 'stats'):
            all_stats["semantic"] = dict(self.semantic_scorer.stats)
        
        if self.judge_scorer and hasattr(self.judge_scorer, 'stats'):
            all_stats["judge"] = dict(self.judge_scorer.stats)
        
        return all_stats


# Factory function
def create_scorer(
    scorer_type: str = "composite",
    llm_client: Optional[LLMClient] = None,
    **kwargs
) -> Union[ExactMatchScorer, RegexScorer, SemanticSimilarityScorer, JudgeBasedScorer, CompositeScorer]:
    """
    Factory function to create scorers.
    
    Args:
        scorer_type: Type of scorer to create
        llm_client: LLM client for judge-based scoring
        **kwargs: Additional scorer configuration
        
    Returns:
        Configured scorer instance
    """
    if scorer_type == "exact_match":
        return ExactMatchScorer()
    elif scorer_type == "regex":
        return RegexScorer()
    elif scorer_type == "semantic":
        return SemanticSimilarityScorer(**kwargs)
    elif scorer_type == "judge":
        if not llm_client:
            raise ValueError("LLM client required for judge-based scorer")
        return JudgeBasedScorer(llm_client, **kwargs)
    elif scorer_type == "composite":
        return CompositeScorer(llm_client, **kwargs)
    else:
        raise ValueError(f"Unknown scorer type: {scorer_type}")


# CLI for testing scoring functionality
async def main():
    """Test scoring functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Answer Scoring")
    parser.add_argument("--answer", required=True, help="Answer to score")
    parser.add_argument("--expected", help="Expected answer for exact match")
    parser.add_argument("--pattern", help="Regex pattern to match")
    parser.add_argument("--reference", help="Reference answer for semantic similarity")
    parser.add_argument("--scorer", default="composite", help="Scorer type to use")
    args = parser.parse_args()
    
    # Create scoring criteria
    criteria = []
    
    if args.expected:
        criteria.append(ScoringCriteria(
            question_id="test",
            score_type=ScoreType.EXACT_MATCH,
            expected_values=[args.expected]
        ))
    
    if args.pattern:
        criteria.append(ScoringCriteria(
            question_id="test", 
            score_type=ScoreType.REGEX_MATCH,
            regex_patterns=[args.pattern]
        ))
    
    if args.reference:
        criteria.append(ScoringCriteria(
            question_id="test",
            score_type=ScoreType.SEMANTIC_SIMILARITY,
            reference_answer=args.reference
        ))
    
    if not criteria:
        print("No scoring criteria provided. Use --expected, --pattern, or --reference")
        return
    
    # Create scorer
    try:
        scorer = create_scorer(args.scorer)
        
        # Score answer
        if hasattr(scorer, 'score') and not asyncio.iscoroutinefunction(scorer.score):
            # Synchronous scorer
            results = {}
            for criterion in criteria:
                result = scorer.score(args.answer, criterion)
                results[criterion.score_type.value] = result
        else:
            # Asynchronous scorer
            results = await scorer.score(args.answer, criteria, "Test question")
        
        # Display results
        print(f"\nScoring Results for: '{args.answer}'\n" + "="*50)
        
        for key, result in results.items():
            print(f"\n{result.score_type.value.upper()}:")
            print(f"  Score: {result.score:.3f}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Explanation: {result.explanation}")
            
            if result.match_details:
                print(f"  Details: {result.match_details}")
        
        # Show composite score if available
        if len(results) > 1 and hasattr(scorer, 'calculate_composite_score'):
            composite = scorer.calculate_composite_score(results)
            print(f"\nCOMPOSITE SCORE:")
            print(f"  Score: {composite.score:.3f}")
            print(f"  Confidence: {composite.confidence:.3f}")
            print(f"  Explanation: {composite.explanation}")
        
    except Exception as e:
        print(f"Scoring test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
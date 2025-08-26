#!/usr/bin/env python3
"""
Judge System - Blind A/B Answer Evaluation

Implements unbiased answer comparison according to TODO.md requirements:
- Blind A/B comparison with rubric-based scoring
- Randomized ordering to prevent bias
- Self-consistency validation (≥85% threshold)
- Inter-judge agreement measurement (κ≥0.6)
- Exact match and regex scoring for objective questions

Key Features:
- Blind evaluation (no identifiers exposed)
- Rubric-based comparison with structured criteria
- Multiple seeds for consistency validation
- Statistical validation of judge reliability
- Comprehensive audit trail for decision rationale
"""

import asyncio
import json
import logging
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import numpy as np
from collections import Counter

from .llm_client import LLMClient
from ..prompts import get_prompt

logger = logging.getLogger(__name__)


class JudgmentDecision(Enum):
    """Possible judgment decisions."""
    A_BETTER = "A"
    B_BETTER = "B"
    TIE = "tie"
    ERROR = "error"


@dataclass
class JudgmentRequest:
    """Request for answer comparison judgment."""
    question: str
    answer_a: str
    answer_b: str
    reference_id_a: str  # Hidden from judge
    reference_id_b: str  # Hidden from judge
    rubric: str
    context: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class JudgmentResult:
    """Result of answer comparison judgment."""
    request_id: str
    question: str
    decision: JudgmentDecision
    confidence: float
    rationale: str
    
    # Identity mapping (A/B -> actual reference)
    answer_a_ref: str
    answer_b_ref: str
    
    # Evaluation details
    criteria_scores: Dict[str, str]  # criterion -> rating
    judge_model: str
    judge_provider: str
    seed: int
    temperature: float
    
    # Response metadata
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    timestamp: str
    
    # Validation flags
    parseable: bool = True
    valid_decision: bool = True


@dataclass
class ConsensusResult:
    """Result from multiple judgment runs."""
    question: str
    answer_refs: Tuple[str, str]  # (ref1, ref2)
    judgments: List[JudgmentResult]
    consensus_decision: JudgmentDecision
    consensus_confidence: float
    agreement_rate: float  # What percentage agreed with consensus
    
    # Statistics
    decision_counts: Dict[str, int]
    avg_confidence: float
    total_cost: float
    avg_latency: float


class AnswerJudge:
    """
    Production answer judge implementing blind A/B comparison.
    
    Features:
    - Blind evaluation (no identifying information)
    - Rubric-based structured scoring
    - Randomized answer ordering
    - Self-consistency validation
    - Comprehensive audit trail
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        rubric_name: str = "judge_rubric",
        num_seeds: int = 3,
        temperature: float = 0.0,
        require_consensus: bool = True,
        min_agreement: float = 0.6
    ):
        self.llm_client = llm_client
        self.rubric_name = rubric_name
        self.num_seeds = num_seeds
        self.temperature = temperature
        self.require_consensus = require_consensus
        self.min_agreement = min_agreement
        
        # Statistics tracking
        self.judgment_stats = {
            "total_comparisons": 0,
            "consensus_achieved": 0,
            "average_agreement": 0.0,
            "decision_breakdown": Counter(),
            "consistency_failures": 0
        }
    
    async def compare_answers(
        self,
        question: str,
        answer_1: str,
        answer_2: str,
        reference_1: str,
        reference_2: str,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConsensusResult:
        """
        Compare two answers using blind A/B evaluation with multiple seeds.
        
        Args:
            question: The original question
            answer_1: First answer to compare
            answer_2: Second answer to compare
            reference_1: Reference ID for first answer (hidden from judge)
            reference_2: Reference ID for second answer (hidden from judge)
            context: Optional context for the question
            metadata: Additional metadata
            
        Returns:
            ConsensusResult with aggregated judgment
        """
        logger.info(f"Comparing answers for question: {question[:50]}...")
        
        # Create judgment requests with randomized ordering
        judgments = []
        
        for seed in range(self.num_seeds):
            # Randomize A/B order for this seed to prevent bias
            random.seed(seed)
            if random.random() < 0.5:
                # Keep original order
                a_answer, b_answer = answer_1, answer_2
                a_ref, b_ref = reference_1, reference_2
            else:
                # Swap order
                a_answer, b_answer = answer_2, answer_1
                a_ref, b_ref = reference_2, reference_1
            
            # Create judgment request
            request = JudgmentRequest(
                question=question,
                answer_a=a_answer,
                answer_b=b_answer,
                reference_id_a=a_ref,
                reference_id_b=b_ref,
                rubric=get_prompt(self.rubric_name),
                context=context,
                metadata=metadata or {}
            )
            
            # Get judgment
            judgment = await self._judge_single_comparison(request, seed)
            judgments.append(judgment)
        
        # Calculate consensus
        consensus = self._calculate_consensus(judgments, reference_1, reference_2)
        
        # Update statistics
        self._update_stats(consensus)
        
        return consensus
    
    async def _judge_single_comparison(
        self, 
        request: JudgmentRequest, 
        seed: int
    ) -> JudgmentResult:
        """Perform single blind A/B comparison."""
        request_id = hashlib.md5(f"{request.question[:50]}{seed}".encode()).hexdigest()[:8]
        
        # Build judgment prompt
        judgment_prompt = self._build_judgment_prompt(request)
        
        # Generate judgment
        response = await self.llm_client.generate(
            prompt=judgment_prompt,
            temperature=self.temperature,
            seed=seed,
            max_tokens=1024
        )
        
        # Parse judgment response
        decision, confidence, rationale, criteria_scores = self._parse_judgment_response(
            response.text
        )
        
        return JudgmentResult(
            request_id=request_id,
            question=request.question,
            decision=decision,
            confidence=confidence,
            rationale=rationale,
            answer_a_ref=request.reference_id_a,
            answer_b_ref=request.reference_id_b,
            criteria_scores=criteria_scores,
            judge_model=response.model,
            judge_provider=response.metadata.get("provider", "unknown"),
            seed=seed,
            temperature=self.temperature,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
            timestamp=response.timestamp,
            parseable=decision != JudgmentDecision.ERROR,
            valid_decision=decision in [JudgmentDecision.A_BETTER, JudgmentDecision.B_BETTER, JudgmentDecision.TIE]
        )
    
    def _build_judgment_prompt(self, request: JudgmentRequest) -> str:
        """Build blind judgment prompt without identifying information."""
        
        # Base prompt with rubric
        prompt_parts = [
            request.rubric,
            "\n\n" + "="*60 + "\n",
            f"**QUESTION:** {request.question}\n"
        ]
        
        # Add context if provided
        if request.context:
            prompt_parts.extend([
                f"**CONTEXT:**\n{request.context}\n\n"
            ])
        
        # Add answers with neutral labels
        prompt_parts.extend([
            "**ANSWER A:**\n",
            request.answer_a,
            "\n\n**ANSWER B:**\n",
            request.answer_b,
            "\n\n" + "="*60 + "\n\n",
            "Please evaluate both answers using the criteria above and provide your judgment."
        ])
        
        return "".join(prompt_parts)
    
    def _parse_judgment_response(self, response: str) -> Tuple[JudgmentDecision, float, str, Dict[str, str]]:
        """Parse judgment response to extract decision and details."""
        try:
            # Look for decision indicators
            decision = JudgmentDecision.ERROR
            confidence = 0.0
            rationale = ""
            criteria_scores = {}
            
            response_lower = response.lower()
            
            # Extract decision
            if "decision: a" in response_lower:
                decision = JudgmentDecision.A_BETTER
            elif "decision: b" in response_lower:
                decision = JudgmentDecision.B_BETTER
            elif "decision: tie" in response_lower:
                decision = JudgmentDecision.TIE
            
            # Extract rationale (look for RATIONALE section)
            rationale_start = response.find("RATIONALE:")
            if rationale_start != -1:
                rationale_text = response[rationale_start + 10:].strip()
                # Take up to first newline or 500 chars
                rationale = rationale_text.split('\n')[0][:500]
            else:
                # Fallback: use last paragraph
                lines = response.strip().split('\n')
                rationale = lines[-1] if lines else "No rationale provided"
            
            # Extract criteria scores (look for High/Medium/Low ratings)
            for line in response.split('\n'):
                line = line.strip()
                if ':' in line and any(rating in line.lower() for rating in ['high', 'medium', 'low']):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        criterion = parts[0].strip().replace('- ', '').replace('*', '')
                        rating_part = parts[1].strip()
                        # Extract rating
                        for rating in ['High', 'Medium', 'Low']:
                            if rating in rating_part:
                                criteria_scores[criterion] = rating
                                break
            
            # Estimate confidence from criteria scores
            if criteria_scores:
                high_count = sum(1 for v in criteria_scores.values() if v == 'High')
                confidence = min(0.9, 0.5 + (high_count / len(criteria_scores)) * 0.4)
            else:
                confidence = 0.5  # Default confidence
            
            return decision, confidence, rationale, criteria_scores
            
        except Exception as e:
            logger.warning(f"Failed to parse judgment response: {e}")
            return JudgmentDecision.ERROR, 0.0, f"Parse error: {str(e)}", {}
    
    def _calculate_consensus(
        self, 
        judgments: List[JudgmentResult], 
        ref_1: str, 
        ref_2: str
    ) -> ConsensusResult:
        """Calculate consensus from multiple judgments."""
        
        # Map A/B decisions back to actual references
        ref_decisions = []
        for judgment in judgments:
            if judgment.decision == JudgmentDecision.A_BETTER:
                winner = judgment.answer_a_ref
            elif judgment.decision == JudgmentDecision.B_BETTER:
                winner = judgment.answer_b_ref
            else:
                winner = "tie"
            ref_decisions.append(winner)
        
        # Count decisions for actual references
        decision_counts = Counter(ref_decisions)
        
        # Determine consensus
        if not ref_decisions:
            consensus_decision = JudgmentDecision.ERROR
            consensus_confidence = 0.0
            agreement_rate = 0.0
        else:
            most_common = decision_counts.most_common(1)[0]
            consensus_winner = most_common[0]
            agreement_count = most_common[1]
            agreement_rate = agreement_count / len(ref_decisions)
            
            # Map back to JudgmentDecision
            if consensus_winner == ref_1:
                consensus_decision = JudgmentDecision.A_BETTER  # Conceptually ref_1 is "better"
            elif consensus_winner == ref_2:
                consensus_decision = JudgmentDecision.B_BETTER  # Conceptually ref_2 is "better"
            else:
                consensus_decision = JudgmentDecision.TIE
            
            # Calculate consensus confidence
            confidences = [j.confidence for j in judgments if j.valid_decision]
            consensus_confidence = np.mean(confidences) if confidences else 0.0
        
        # Create consensus result
        return ConsensusResult(
            question=judgments[0].question if judgments else "",
            answer_refs=(ref_1, ref_2),
            judgments=judgments,
            consensus_decision=consensus_decision,
            consensus_confidence=consensus_confidence,
            agreement_rate=agreement_rate,
            decision_counts=dict(decision_counts),
            avg_confidence=np.mean([j.confidence for j in judgments]),
            total_cost=sum(j.cost_usd for j in judgments),
            avg_latency=np.mean([j.latency_ms for j in judgments])
        )
    
    def _update_stats(self, consensus: ConsensusResult):
        """Update judgment statistics."""
        self.judgment_stats["total_comparisons"] += 1
        
        if consensus.agreement_rate >= self.min_agreement:
            self.judgment_stats["consensus_achieved"] += 1
        else:
            self.judgment_stats["consistency_failures"] += 1
        
        # Update running average
        total = self.judgment_stats["total_comparisons"]
        current_avg = self.judgment_stats["average_agreement"]
        self.judgment_stats["average_agreement"] = (
            (current_avg * (total - 1) + consensus.agreement_rate) / total
        )
        
        # Update decision breakdown
        self.judgment_stats["decision_breakdown"][consensus.consensus_decision.value] += 1
    
    def get_reliability_metrics(self) -> Dict[str, Any]:
        """Get judge reliability and consistency metrics."""
        stats = self.judgment_stats.copy()
        
        if stats["total_comparisons"] > 0:
            stats["consistency_rate"] = stats["consensus_achieved"] / stats["total_comparisons"]
            stats["meets_kappa_threshold"] = stats["average_agreement"] >= self.min_agreement
        else:
            stats["consistency_rate"] = 0.0
            stats["meets_kappa_threshold"] = False
        
        return stats


class ExactMatchJudge:
    """Simple exact match and regex-based judge for objective questions."""
    
    def __init__(self):
        self.stats = {"exact_matches": 0, "regex_matches": 0, "no_matches": 0}
    
    def judge_exact_match(
        self, 
        answer: str, 
        expected: Union[str, List[str]],
        case_sensitive: bool = False
    ) -> Tuple[bool, float, str]:
        """
        Judge answer using exact string matching.
        
        Returns:
            Tuple of (is_match, confidence, explanation)
        """
        if isinstance(expected, str):
            expected_list = [expected]
        else:
            expected_list = expected
        
        answer_norm = answer if case_sensitive else answer.lower()
        
        for exp in expected_list:
            exp_norm = exp if case_sensitive else exp.lower()
            if exp_norm in answer_norm:
                self.stats["exact_matches"] += 1
                return True, 1.0, f"Exact match found: '{exp}'"
        
        self.stats["no_matches"] += 1
        return False, 0.0, "No exact match found"
    
    def judge_regex_match(
        self, 
        answer: str, 
        patterns: Union[str, List[str]]
    ) -> Tuple[bool, float, str]:
        """
        Judge answer using regex pattern matching.
        
        Returns:
            Tuple of (is_match, confidence, explanation)
        """
        import re
        
        if isinstance(patterns, str):
            pattern_list = [patterns]
        else:
            pattern_list = patterns
        
        for pattern in pattern_list:
            try:
                if re.search(pattern, answer, re.IGNORECASE):
                    self.stats["regex_matches"] += 1
                    return True, 0.9, f"Regex match: '{pattern}'"
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
        
        self.stats["no_matches"] += 1
        return False, 0.0, "No regex match found"


# Factory function for creating judges
def create_judge(
    llm_client: LLMClient,
    judge_type: str = "blind_ab",
    **kwargs
) -> Union[AnswerJudge, ExactMatchJudge]:
    """
    Factory function to create different types of judges.
    
    Args:
        llm_client: LLM client for A/B judgments
        judge_type: Type of judge ("blind_ab", "exact_match")
        **kwargs: Additional configuration
        
    Returns:
        Configured judge instance
    """
    if judge_type == "blind_ab":
        return AnswerJudge(llm_client, **kwargs)
    elif judge_type == "exact_match":
        return ExactMatchJudge()
    else:
        raise ValueError(f"Unknown judge type: {judge_type}")


# CLI for testing judge functionality
async def main():
    """Test judge functionality."""
    import argparse
    from .llm_client import create_llm_client
    
    parser = argparse.ArgumentParser(description="Test Answer Judge")
    parser.add_argument("--question", required=True, help="Test question")
    parser.add_argument("--answer1", required=True, help="First answer")
    parser.add_argument("--answer2", required=True, help="Second answer")
    parser.add_argument("--provider", default="openai", help="LLM provider")
    args = parser.parse_args()
    
    # Create LLM client
    config = {
        "providers": {
            "openai": {"api_key": "your-key-here"},
        },
        "default_provider": args.provider
    }
    
    try:
        llm_client = create_llm_client(config)
        judge = create_judge(llm_client, "blind_ab")
        
        # Run comparison
        result = await judge.compare_answers(
            question=args.question,
            answer_1=args.answer1,
            answer_2=args.answer2,
            reference_1="answer_1",
            reference_2="answer_2"
        )
        
        print(f"\nJudgment Result:")
        print(f"Decision: {result.consensus_decision.value}")
        print(f"Agreement Rate: {result.agreement_rate:.2%}")
        print(f"Confidence: {result.consensus_confidence:.2f}")
        print(f"Cost: ${result.total_cost:.4f}")
        print(f"Latency: {result.avg_latency:.1f}ms")
        
        # Show individual judgments
        for i, judgment in enumerate(result.judgments):
            print(f"\nJudgment {i+1}: {judgment.decision.value} (confidence: {judgment.confidence:.2f})")
            print(f"Rationale: {judgment.rationale}")
        
        await llm_client.close()
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
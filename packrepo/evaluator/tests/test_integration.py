#!/usr/bin/env python3
"""
Integration Tests for PackRepo QA Harness

Tests the complete end-to-end QA evaluation pipeline including:
- LLM client integration
- QA runner with prompt management
- Judge system with blind A/B comparison
- Scoring system with multiple methods
- Complete evaluation workflow
"""

import asyncio
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import sys
sys.path.append(str(Path(__file__).parent.parent))

from harness.llm_client import LLMClient, LLMResponse, create_llm_client
from harness.runner import QARunner, QARunConfig, QATask
from harness.judge import AnswerJudge, JudgmentDecision
from harness.scorers import CompositeScorer, ScoringCriteria, ScoreType


class TestEndToEndWorkflow:
    """Test complete QA evaluation workflow."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client for testing."""
        client = Mock(spec=LLMClient)
        
        # Mock different responses for different prompts
        def mock_generate(prompt, **kwargs):
            if "What is the main purpose" in prompt:
                response_text = "PackRepo is a repository packing system that optimizes content for LLM consumption with token budgets and deterministic output."
            elif "ANSWER A:" in prompt and "ANSWER B:" in prompt:
                # Judge prompt
                response_text = """
EVALUATION:

Answer A Analysis:
- Accuracy: High - Technical claims are accurate
- Completeness: High - Addresses all parts of question  
- Evidence: High - Cites specific code references
- Clarity: High - Well-organized and readable
- Calibration: High - Confidence appropriate

Answer B Analysis:
- Accuracy: Medium - Some technical claims unclear
- Completeness: Medium - Partially addresses question
- Evidence: Low - Few specific references
- Clarity: Medium - Somewhat organized
- Calibration: Medium - Confidence level appropriate

DECISION: A

RATIONALE: Answer A provides more accurate technical details with better evidence and completeness.
"""
            else:
                response_text = "This is a test response to the question."
            
            return asyncio.coroutine(lambda: LLMResponse(
                text=response_text,
                model="test-model",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost_usd=0.001,
                latency_ms=500.0,
                timestamp="2024-01-01T00:00:00Z",
                request_id="test-123",
                metadata={"provider": "test"}
            ))()
        
        client.generate.side_effect = mock_generate
        client.close.return_value = asyncio.coroutine(lambda: None)()
        
        return client
    
    @pytest.fixture
    def sample_pack_content(self):
        """Sample pack content for testing."""
        return """
# PackRepo - Repository Packing System

## Overview
PackRepo is a sophisticated system for packing git repositories into LLM-optimized formats.
The system enforces token budgets and provides deterministic, reproducible output.

## Architecture
- **Chunker**: Processes code files using tree-sitter for language-specific parsing
- **Selector**: Uses facility location and MMR algorithms for optimal content selection  
- **Tokenizer**: Counts tokens using tiktoken with downstream model compatibility
- **PackFmt**: Formats output with JSON index and structured body sections

## Key Features
- Token budget enforcement with <0.5% underflow tolerance
- Deterministic output with `--no-llm` flag for reproducible results
- Oracle-based validation for contracts and properties
- Metamorphic testing for robustness validation

## Selection Algorithms
The system uses submodular optimization with facility location objectives:
- Greedy selection for coverage maximization
- MMR (Maximal Marginal Relevance) for redundancy reduction
- Budget-constrained optimization with overflow prevention
"""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            
            # Create pack files
            pack_dir = workspace / "packs"
            pack_dir.mkdir()
            
            # Create sample pack content
            pack_content = {
                "index": {
                    "repository": "test-repo",
                    "total_tokens": 1000,
                    "chunks": [
                        {"file": "README.md", "start_line": 1, "end_line": 50, "tokens": 500},
                        {"file": "src/main.py", "start_line": 1, "end_line": 100, "tokens": 500}
                    ]
                },
                "body": "Sample pack body content with repository information and code sections."
            }
            
            for variant in ["V0_baseline", "V1_deterministic"]:
                pack_file = pack_dir / f"{variant}.pack"
                pack_file.write_text(json.dumps(pack_content))
            
            yield workspace
    
    @pytest.mark.asyncio
    async def test_complete_qa_workflow(self, mock_llm_client, temp_workspace):
        """Test complete QA evaluation workflow."""
        
        # Create QA configuration
        config = QARunConfig(
            pack_paths={
                "V0_baseline": temp_workspace / "packs" / "V0_baseline.pack",
                "V1_deterministic": temp_workspace / "packs" / "V1_deterministic.pack"
            },
            tasks=[
                QATask(
                    question_id="purpose",
                    question="What is the main purpose of this repository?",
                    context_budget=1000,
                    difficulty="easy",
                    category="overview"
                ),
                QATask(
                    question_id="architecture", 
                    question="What are the key architectural components?",
                    context_budget=1200,
                    difficulty="medium",
                    category="technical"
                )
            ],
            llm_config={
                "providers": {"test": {}},
                "default_provider": "test"
            },
            seeds=[0, 1],
            temperature=0.0,
            output_dir=temp_workspace / "qa_outputs"
        )
        
        # Create runner with mocked client
        runner = QARunner(config)
        runner.llm_client = mock_llm_client
        
        # Run evaluation
        results = await runner.run_evaluation()
        
        # Validate results
        assert results["overall_stats"]["total_evaluations"] > 0
        assert results["overall_stats"]["success_rate"] > 0
        assert len(results["variant_stats"]) == 2
        assert "V0_baseline" in results["variant_stats"]
        assert "V1_deterministic" in results["variant_stats"]
        
        # Check output files were created
        output_dir = temp_workspace / "qa_outputs"
        assert (output_dir / "qa_answers.jsonl").exists()
        assert (output_dir / "qa_summary.json").exists()
        assert (output_dir / "qa_raw_results.json").exists()
    
    @pytest.mark.asyncio
    async def test_judge_comparison_workflow(self, mock_llm_client):
        """Test blind A/B judge comparison."""
        judge = AnswerJudge(
            llm_client=mock_llm_client,
            num_seeds=2,
            temperature=0.0
        )
        
        answer_1 = "PackRepo is a comprehensive repository packing system with token budget enforcement."
        answer_2 = "It's a tool for processing repositories."
        
        result = await judge.compare_answers(
            question="What is PackRepo?",
            answer_1=answer_1,
            answer_2=answer_2,
            reference_1="detailed_answer",
            reference_2="brief_answer"
        )
        
        # Validate judgment
        assert result.consensus_decision in [JudgmentDecision.A_BETTER, JudgmentDecision.B_BETTER, JudgmentDecision.TIE]
        assert 0.0 <= result.consensus_confidence <= 1.0
        assert len(result.judgments) == 2
        assert result.total_cost > 0
    
    def test_scoring_workflow(self):
        """Test multi-method scoring workflow."""
        scorer = CompositeScorer(llm_client=None)  # No LLM for basic scoring
        
        # Test answer
        answer = "PackRepo uses chunker, selector, tokenizer, and packfmt components for repository packing."
        
        # Create scoring criteria
        criteria = [
            ScoringCriteria(
                question_id="architecture",
                score_type=ScoreType.EXACT_MATCH,
                expected_values=["chunker", "selector", "tokenizer"],
                weight=0.4
            ),
            ScoringCriteria(
                question_id="architecture", 
                score_type=ScoreType.REGEX_MATCH,
                regex_patterns=[r"packfmt", r"repository.*pack"],
                weight=0.3
            )
        ]
        
        # Score synchronously
        results = {}
        for criterion in criteria:
            if criterion.score_type == ScoreType.EXACT_MATCH:
                result = scorer.exact_scorer.score(answer, criterion)
                results[f"exact_{criterion.question_id}"] = result
            elif criterion.score_type == ScoreType.REGEX_MATCH:
                result = scorer.regex_scorer.score(answer, criterion)
                results[f"regex_{criterion.question_id}"] = result
        
        # Validate scoring results
        assert len(results) == 2
        for result in results.values():
            assert 0.0 <= result.score <= 1.0
            assert 0.0 <= result.confidence <= 1.0
            assert len(result.explanation) > 0
        
        # Test composite scoring
        composite = scorer.calculate_composite_score(results)
        assert 0.0 <= composite.score <= 1.0
        assert composite.score_type == ScoreType.COMPOSITE


class TestErrorHandling:
    """Test error handling and graceful degradation."""
    
    @pytest.mark.asyncio
    async def test_llm_client_failure_handling(self):
        """Test handling of LLM client failures."""
        # Create client that fails
        failing_client = Mock(spec=LLMClient)
        failing_client.generate.side_effect = Exception("API Error")
        failing_client.close.return_value = asyncio.coroutine(lambda: None)()
        
        judge = AnswerJudge(failing_client, num_seeds=1)
        
        # Should handle failure gracefully
        with pytest.raises(Exception):
            await judge.compare_answers(
                question="Test?",
                answer_1="A1",
                answer_2="A2", 
                reference_1="ref1",
                reference_2="ref2"
            )
    
    def test_invalid_pack_file_handling(self, tmp_path):
        """Test handling of invalid pack files."""
        # Create invalid pack file
        invalid_pack = tmp_path / "invalid.pack"
        invalid_pack.write_text("invalid json content")
        
        config = QARunConfig(
            pack_paths={"invalid": invalid_pack},
            tasks=[
                QATask(
                    question_id="test",
                    question="Test question?",
                    context_budget=1000
                )
            ],
            llm_config={
                "providers": {"test": {}},
                "default_provider": "test"
            },
            seeds=[0],
            output_dir=tmp_path / "output"
        )
        
        runner = QARunner(config)
        
        # Should handle gracefully (content will be empty string)
        content = runner._load_pack_content(invalid_pack)
        assert content == ""  # Should return empty string for invalid JSON


class TestStatisticalValidation:
    """Test statistical validation and consistency checking."""
    
    def test_judge_reliability_metrics(self):
        """Test judge reliability calculation."""
        mock_client = Mock(spec=LLMClient)
        judge = AnswerJudge(mock_client, min_agreement=0.6)
        
        # Simulate judgments with good agreement
        from harness.judge import ConsensusResult, JudgmentResult
        
        consensus = ConsensusResult(
            question="Test",
            answer_refs=("A", "B"),
            judgments=[],
            consensus_decision=JudgmentDecision.A_BETTER,
            consensus_confidence=0.8,
            agreement_rate=0.8,  # Above threshold
            decision_counts={"A": 2, "B": 0, "tie": 1},
            avg_confidence=0.75,
            total_cost=0.006,
            avg_latency=400.0
        )
        
        judge._update_stats(consensus)
        
        metrics = judge.get_reliability_metrics()
        assert metrics["total_comparisons"] == 1
        assert metrics["consensus_achieved"] == 1
        assert metrics["average_agreement"] == 0.8
        assert metrics["meets_kappa_threshold"] == True
    
    def test_score_confidence_calculation(self):
        """Test confidence calculation in scoring."""
        from harness.scorers import ExactMatchScorer, ScoringCriteria, ScoreType
        
        scorer = ExactMatchScorer()
        
        # Perfect match should have high confidence
        criteria = ScoringCriteria(
            question_id="test",
            score_type=ScoreType.EXACT_MATCH,
            expected_values=["test", "match"]
        )
        
        result = scorer.score("This is a test match example", criteria)
        assert result.score == 1.0  # Found both expected values
        assert result.confidence == 1.0  # Perfect confidence for exact match


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
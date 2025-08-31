#!/usr/bin/env python3
"""
Main evaluation runner for FastPath variants (baseline + V1→V5).

Implements the run matrix from TODO.md with support for:
- Paired evaluation with consistent seeds/repos/questions
- Budget parity enforcement (50k/120k/200k tokens)
- Negative controls (scramble, flip, random_quota)
- Feature flag-driven variant selection
- JSONL output for downstream analysis
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import hashlib
import random
import numpy as np

# Import PackRepo and FastPath components
try:
    from packrepo.fastpath import FastScanner, HeuristicScorer, TTLScheduler
    from packrepo.fastpath.feature_flags import FeatureFlags
    from packrepo.fastpath.integrated_v5 import FastPathV5System
    from packrepo.evaluator.matrix_runner import MatrixRunner
    from packrepo.evaluator.harness.runner import EvaluationRunner
    from packrepo.evaluator.statistics.paired_analysis import PairedAnalysis
except ImportError as e:
    print(f"Error importing PackRepo components: {e}")
    print("Please ensure PackRepo is properly installed and PYTHONPATH is set")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    system: str  # baseline, v1, v2, v3, v4, v5, or negative control
    budgets: List[int]  # Token budgets (50k, 120k, 200k)
    paired: bool = True  # Use paired evaluation
    seeds: int = 100  # Number of evaluation pairs
    output_file: Optional[Path] = None
    
    # Feature flags for variants
    policy_v2: bool = False
    centrality: bool = False
    demote: bool = False
    patch: bool = False
    router: bool = False
    
    # Negative controls
    neg_ctrl: Optional[str] = None  # scramble, flip, random_quota


@dataclass
class EvaluationResult:
    """Single evaluation result."""
    system: str
    budget: int
    seed: int
    repo_name: str
    question_id: str
    
    # Metrics
    qa_score: float
    tokens_used: int
    latency_ms: float
    memory_mb: float
    
    # Metadata
    timestamp: str
    selection_hash: str
    features: Dict[str, Any]
    
    def to_jsonl(self) -> str:
        """Convert to JSONL format."""
        return json.dumps(asdict(self))


class FastPathEvaluator:
    """Main evaluator for FastPath variants."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.evaluation_runner = EvaluationRunner()
        self.paired_analysis = PairedAnalysis()
        
    def setup_feature_flags(self) -> FeatureFlags:
        """Setup feature flags based on system variant."""
        flags = FeatureFlags()
        
        if self.config.system == "baseline":
            # All flags off for baseline
            flags.policy_v2 = False
            flags.centrality = False
            flags.demote = False
            flags.patch = False
            flags.router = False
            
        elif self.config.system == "v1":
            flags.policy_v2 = True
            flags.centrality = False
            flags.demote = False
            flags.patch = False
            flags.router = False
            
        elif self.config.system == "v2":
            flags.policy_v2 = True
            flags.centrality = True
            flags.demote = False
            flags.patch = False
            flags.router = False
            
        elif self.config.system == "v3":
            flags.policy_v2 = True
            flags.centrality = True
            flags.demote = True
            flags.patch = False
            flags.router = False
            
        elif self.config.system == "v4":
            flags.policy_v2 = True
            flags.centrality = True
            flags.demote = True
            flags.patch = True
            flags.router = False
            
        elif self.config.system == "v5":
            flags.policy_v2 = True
            flags.centrality = True
            flags.demote = True
            flags.patch = True
            flags.router = True
            
        # Apply negative controls
        if self.config.neg_ctrl:
            flags.neg_ctrl = self.config.neg_ctrl
            
        return flags
        
    def run_evaluation(self) -> List[EvaluationResult]:
        """Run complete evaluation for the configured system."""
        flags = self.setup_feature_flags()
        results = []
        
        logger.info(f"Starting evaluation for system: {self.config.system}")
        logger.info(f"Budgets: {self.config.budgets}")
        logger.info(f"Seeds: {self.config.seeds}")
        
        # Create FastPath V5 system with appropriate flags
        fastpath_system = FastPathV5System(feature_flags=flags)
        
        for budget in self.config.budgets:
            for seed in range(self.config.seeds):
                try:
                    result = self._evaluate_single(
                        fastpath_system, budget, seed
                    )
                    results.append(result)
                    
                    if seed % 10 == 0:
                        logger.info(f"Completed {seed+1}/{self.config.seeds} evaluations for budget {budget}")
                        
                except Exception as e:
                    logger.error(f"Error in evaluation (budget={budget}, seed={seed}): {e}")
                    continue
                    
        logger.info(f"Completed evaluation with {len(results)} results")
        return results
        
    def _evaluate_single(
        self, 
        fastpath_system: FastPathV5System,
        budget: int,
        seed: int
    ) -> EvaluationResult:
        """Evaluate a single configuration."""
        # Set reproducible seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Get test repository and question (paired evaluation)
        repo_name, question_data = self._get_paired_test_case(seed)
        
        start_time = time.time()
        
        # Run FastPath packing with budget constraint
        pack_result = fastpath_system.pack_repository(
            repo_path=f"test_repos/{repo_name}",
            token_budget=budget,
            seed=seed
        )
        
        # Evaluate QA performance
        qa_result = self.evaluation_runner.evaluate_qa(
            packed_repo=pack_result.packed_content,
            question=question_data['question'],
            expected_answer=question_data['expected_answer']
        )
        
        end_time = time.time()
        
        # Calculate selection hash for determinism checking
        selection_hash = hashlib.sha256(
            json.dumps(pack_result.selected_files, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        return EvaluationResult(
            system=self.config.system,
            budget=budget,
            seed=seed,
            repo_name=repo_name,
            question_id=question_data['id'],
            qa_score=qa_result.score,
            tokens_used=pack_result.actual_tokens,
            latency_ms=(end_time - start_time) * 1000,
            memory_mb=pack_result.memory_usage_mb,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            selection_hash=selection_hash,
            features=pack_result.features
        )
        
    def _get_paired_test_case(self, seed: int) -> tuple[str, Dict[str, Any]]:
        """Get consistent test case for paired evaluation."""
        # Use seed to select consistent repo and question
        repos = [
            "test_repo_python_small",
            "test_repo_javascript_medium", 
            "test_repo_rust_large",
            "test_repo_typescript_small",
            "test_repo_go_medium"
        ]
        
        repo_idx = seed % len(repos)
        repo_name = repos[repo_idx]
        
        # Generate consistent question for this repo+seed combination
        question_data = {
            'id': f"{repo_name}_q{seed}",
            'question': f"How does the main functionality work in {repo_name}?",
            'expected_answer': "Implementation details should be extractable from packed content"
        }
        
        return repo_name, question_data
        
    def save_results(self, results: List[EvaluationResult], output_file: Path):
        """Save results in JSONL format."""
        with open(output_file, 'w') as f:
            for result in results:
                f.write(result.to_jsonl() + '\n')
                
        logger.info(f"Saved {len(results)} results to {output_file}")


def parse_budgets(budget_str: str) -> List[int]:
    """Parse budget string like '50k,120k,200k' to list of integers."""
    budgets = []
    for b in budget_str.split(','):
        b = b.strip().lower()
        if b.endswith('k'):
            budgets.append(int(b[:-1]) * 1000)
        else:
            budgets.append(int(b))
    return budgets


def main():
    parser = argparse.ArgumentParser(
        description="FastPath evaluation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline evaluation
  python run.py --system baseline --budgets 50k,120k,200k --paired --seeds 100 --emit artifacts/baseline.jsonl

  # FastPath V5 evaluation  
  python run.py --system v5 --budgets 50k,120k,200k --paired --seeds 100 --emit artifacts/v5.jsonl

  # Negative control - scrambled graph
  python run.py --system scramble --budgets 50k,120k,200k --paired --seeds 50 --emit artifacts/ctrl_scramble.jsonl
        """
    )
    
    parser.add_argument('--system', required=True,
                        choices=['baseline', 'v1', 'v2', 'v3', 'v4', 'v5', 'scramble', 'flip', 'random_quota'],
                        help='System variant to evaluate')
    parser.add_argument('--budgets', required=True, type=str,
                        help='Comma-separated token budgets (e.g. 50k,120k,200k)')
    parser.add_argument('--paired', action='store_true', default=False,
                        help='Use paired evaluation with consistent seeds')
    parser.add_argument('--seeds', type=int, default=100,
                        help='Number of evaluation pairs')
    parser.add_argument('--emit', type=str, required=True,
                        help='Output JSONL file path')
    
    args = parser.parse_args()
    
    # Handle environment variables for feature flags
    if 'FASTPATH_POLICY_V2' in os.environ:
        policy_v2 = os.environ['FASTPATH_POLICY_V2'] == '1'
    else:
        policy_v2 = args.system != 'baseline'
        
    if 'FASTPATH_CENTRALITY' in os.environ:
        centrality = os.environ['FASTPATH_CENTRALITY'] == '1'
    else:
        centrality = args.system in ['v2', 'v3', 'v4', 'v5']
        
    if 'FASTPATH_DEMOTE' in os.environ:
        demote = os.environ['FASTPATH_DEMOTE'] == '1'
    else:
        demote = args.system in ['v3', 'v4', 'v5']
        
    if 'FASTPATH_PATCH' in os.environ:
        patch = os.environ['FASTPATH_PATCH'] == '1'
    else:
        patch = args.system in ['v4', 'v5']
        
    if 'FASTPATH_ROUTER' in os.environ:
        router = os.environ['FASTPATH_ROUTER'] == '1'
    else:
        router = args.system == 'v5'
        
    # Handle negative controls
    neg_ctrl = None
    if args.system in ['scramble', 'flip', 'random_quota']:
        neg_ctrl = args.system
        args.system = 'v5'  # Use V5 base with negative control
        
    if 'FASTPATH_NEGCTRL' in os.environ:
        neg_ctrl = os.environ['FASTPATH_NEGCTRL']
    
    # Create configuration
    config = BenchmarkConfig(
        system=args.system,
        budgets=parse_budgets(args.budgets),
        paired=args.paired,
        seeds=args.seeds,
        output_file=Path(args.emit),
        policy_v2=policy_v2,
        centrality=centrality,
        demote=demote,
        patch=patch,
        router=router,
        neg_ctrl=neg_ctrl
    )
    
    # Run evaluation
    evaluator = FastPathEvaluator(config)
    results = evaluator.run_evaluation()
    
    # Save results
    output_path = Path(args.emit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_results(results, output_path)
    
    print(f"Evaluation complete. Results saved to {output_path}")
    print(f"Total evaluations: {len(results)}")
    
    # Print summary statistics
    if results:
        avg_qa_score = sum(r.qa_score for r in results) / len(results)
        avg_tokens = sum(r.tokens_used for r in results) / len(results)
        avg_latency = sum(r.latency_ms for r in results) / len(results)
        
        print(f"Average QA score: {avg_qa_score:.3f}")
        print(f"Average tokens used: {avg_tokens:.0f}")  
        print(f"Average latency: {avg_latency:.1f} ms")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Artifact consolidation system for FastPath evaluation pipeline.

Consolidates results from multiple JSONL files into unified datasets for analysis:
- Collects evaluation results across all variants (baseline, V1-V5, controls)
- Validates data consistency and completeness  
- Computes summary statistics and metadata
- Generates consolidated JSON output for downstream analysis
- Supports glob patterns for batch collection
"""

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
import numpy as np
import hashlib
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConsolidatedResult:
    """Single consolidated evaluation result."""
    
    # Identifiers
    system: str
    budget: int
    seed: int
    repo_name: str
    question_id: str
    
    # Core metrics
    qa_score: float
    tokens_used: int
    latency_ms: float
    memory_mb: float
    
    # Quality indicators
    selection_hash: str
    timestamp: str
    
    # Optional extended metrics
    features: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None


@dataclass  
class CollectionSummary:
    """Summary statistics for collected results."""
    
    # Data overview
    total_results: int
    unique_systems: List[str]
    budgets: List[int] 
    seeds_per_system: Dict[str, int]
    
    # Quality metrics
    mean_qa_scores: Dict[str, float]
    token_utilization: Dict[str, float]
    latency_p95: Dict[str, float]
    
    # Validation results
    missing_pairs: List[Dict[str, Any]]
    duplicate_entries: List[Dict[str, Any]]
    data_quality_score: float
    
    # Metadata
    collection_timestamp: str
    source_files: List[str]


class ArtifactCollector:
    """Consolidates evaluation artifacts from multiple sources."""
    
    def __init__(self):
        self.results: List[ConsolidatedResult] = []
        self.errors: List[str] = []
        
    def collect_from_pattern(self, pattern: str) -> None:
        """Collect results from files matching glob pattern."""
        files = glob.glob(pattern)
        
        if not files:
            logger.warning(f"No files found matching pattern: {pattern}")
            return
            
        logger.info(f"Collecting from {len(files)} files: {files}")
        
        for file_path in files:
            try:
                self.collect_from_file(Path(file_path))
            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                logger.error(error_msg)
                self.errors.append(error_msg)
                
    def collect_from_file(self, file_path: Path) -> None:
        """Collect results from a single JSONL file."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        logger.info(f"Processing file: {file_path}")
        
        count = 0
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    result = self._parse_result(data, file_path, line_num)
                    if result:
                        self.results.append(result)
                        count += 1
                except Exception as e:
                    error_msg = f"Error parsing line {line_num} in {file_path}: {e}"
                    logger.warning(error_msg)
                    self.errors.append(error_msg)
                    
        logger.info(f"Collected {count} results from {file_path}")
        
    def _parse_result(self, data: Dict[str, Any], source_file: Path, line_num: int) -> Optional[ConsolidatedResult]:
        """Parse a single result record."""
        try:
            # Validate required fields
            required_fields = ['system', 'budget', 'seed', 'qa_score', 'tokens_used']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
                
            # Parse with defaults for optional fields
            return ConsolidatedResult(
                system=data['system'],
                budget=int(data['budget']),
                seed=int(data['seed']),
                repo_name=data.get('repo_name', 'unknown'),
                question_id=data.get('question_id', f'q_{data["seed"]}'),
                qa_score=float(data['qa_score']),
                tokens_used=int(data['tokens_used']),
                latency_ms=float(data.get('latency_ms', 0.0)),
                memory_mb=float(data.get('memory_mb', 0.0)),
                selection_hash=data.get('selection_hash', ''),
                timestamp=data.get('timestamp', ''),
                features=data.get('features'),
                errors=None
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse result from {source_file}:{line_num}: {e}")
            return None
            
    def validate_collection(self) -> CollectionSummary:
        """Validate collected data and generate summary."""
        if not self.results:
            logger.warning("No results collected")
            return self._empty_summary()
            
        logger.info(f"Validating {len(self.results)} collected results")
        
        # Basic statistics
        systems = list(set(r.system for r in self.results))
        budgets = sorted(list(set(r.budget for r in self.results)))
        
        # Count seeds per system
        seeds_per_system = {}
        for system in systems:
            system_results = [r for r in self.results if r.system == system]
            unique_seeds = set((r.budget, r.seed) for r in system_results)
            seeds_per_system[system] = len(unique_seeds)
            
        # Quality metrics
        mean_qa_scores = {}
        token_utilization = {}
        latency_p95 = {}
        
        for system in systems:
            system_results = [r for r in self.results if r.system == system]
            
            qa_scores = [r.qa_score for r in system_results]
            tokens = [r.tokens_used for r in system_results]
            latencies = [r.latency_ms for r in system_results]
            
            mean_qa_scores[system] = np.mean(qa_scores)
            token_utilization[system] = np.mean(tokens)
            latency_p95[system] = np.percentile(latencies, 95) if latencies else 0.0
            
        # Validate paired structure
        missing_pairs = self._find_missing_pairs()
        duplicate_entries = self._find_duplicates()
        
        # Data quality score
        quality_score = self._compute_quality_score(missing_pairs, duplicate_entries)
        
        return CollectionSummary(
            total_results=len(self.results),
            unique_systems=systems,
            budgets=budgets,
            seeds_per_system=seeds_per_system,
            mean_qa_scores=mean_qa_scores,
            token_utilization=token_utilization,
            latency_p95=latency_p95,
            missing_pairs=missing_pairs,
            duplicate_entries=duplicate_entries,
            data_quality_score=quality_score,
            collection_timestamp=str(Path.cwd().stat().st_mtime),
            source_files=list(set(str(f) for f in glob.glob("*.jsonl")))
        )
        
    def _find_missing_pairs(self) -> List[Dict[str, Any]]:
        """Find missing pairs in evaluation data."""
        missing = []
        
        # Group by (budget, seed) to find missing systems
        pairs = defaultdict(set)
        for result in self.results:
            key = (result.budget, result.seed)
            pairs[key].add(result.system)
            
        # Find expected systems (baseline + variants)
        all_systems = set(r.system for r in self.results)
        
        for (budget, seed), systems in pairs.items():
            missing_systems = all_systems - systems
            if missing_systems:
                missing.append({
                    'budget': budget,
                    'seed': seed,
                    'missing_systems': list(missing_systems)
                })
                
        return missing
        
    def _find_duplicates(self) -> List[Dict[str, Any]]:
        """Find duplicate entries."""
        duplicates = []
        seen = set()
        
        for result in self.results:
            key = (result.system, result.budget, result.seed, result.repo_name, result.question_id)
            if key in seen:
                duplicates.append({
                    'system': result.system,
                    'budget': result.budget,
                    'seed': result.seed,
                    'repo_name': result.repo_name,
                    'question_id': result.question_id
                })
            else:
                seen.add(key)
                
        return duplicates
        
    def _compute_quality_score(self, missing_pairs: List, duplicates: List) -> float:
        """Compute overall data quality score (0-100)."""
        if not self.results:
            return 0.0
            
        # Penalize missing pairs and duplicates
        completeness = max(0, 100 - len(missing_pairs) * 5)  # -5 points per missing pair
        uniqueness = max(0, 100 - len(duplicates) * 10)      # -10 points per duplicate
        
        # Check for reasonable QA scores (should be between 0 and 1)
        qa_scores = [r.qa_score for r in self.results]
        valid_qa_scores = sum(1 for score in qa_scores if 0 <= score <= 1)
        qa_validity = (valid_qa_scores / len(qa_scores)) * 100
        
        # Overall quality score (weighted average)
        quality_score = (completeness * 0.4 + uniqueness * 0.3 + qa_validity * 0.3)
        return round(quality_score, 1)
        
    def _empty_summary(self) -> CollectionSummary:
        """Return empty summary for case with no results."""
        return CollectionSummary(
            total_results=0,
            unique_systems=[],
            budgets=[],
            seeds_per_system={},
            mean_qa_scores={},
            token_utilization={},
            latency_p95={},
            missing_pairs=[],
            duplicate_entries=[],
            data_quality_score=0.0,
            collection_timestamp="",
            source_files=[]
        )
        
    def save_consolidated_results(self, output_file: Path) -> None:
        """Save consolidated results with summary."""
        
        # Generate summary
        summary = self.validate_collection()
        
        # Prepare output data
        output_data = {
            'summary': asdict(summary),
            'collection_errors': self.errors,
            'results': [asdict(result) for result in self.results]
        }
        
        # Save to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        logger.info(f"Saved {len(self.results)} consolidated results to {output_file}")
        logger.info(f"Data quality score: {summary.data_quality_score}/100")
        
        # Print validation warnings
        if summary.missing_pairs:
            logger.warning(f"Found {len(summary.missing_pairs)} incomplete evaluation pairs")
            
        if summary.duplicate_entries:
            logger.warning(f"Found {len(summary.duplicate_entries)} duplicate entries")
            
        if self.errors:
            logger.warning(f"Encountered {len(self.errors)} collection errors")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate FastPath evaluation artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect all JSONL files in artifacts directory
  python collect.py --glob "artifacts/*.jsonl" --out artifacts/collected.json
  
  # Collect specific files
  python collect.py --files artifacts/baseline.jsonl artifacts/v5.jsonl --out artifacts/collected.json
        """
    )
    
    parser.add_argument('--glob', type=str,
                        help='Glob pattern for input files (e.g., "artifacts/*.jsonl")')
    parser.add_argument('--files', nargs='+',
                        help='Specific input files')
    parser.add_argument('--out', type=str, required=True,
                        help='Output consolidated JSON file')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate data, do not save results')
    
    args = parser.parse_args()
    
    if not args.glob and not args.files:
        parser.error("Must specify either --glob or --files")
        
    # Create collector
    collector = ArtifactCollector()
    
    # Collect from specified sources
    if args.glob:
        collector.collect_from_pattern(args.glob)
        
    if args.files:
        for file_path in args.files:
            collector.collect_from_file(Path(file_path))
            
    if not collector.results and not args.validate_only:
        logger.error("No results collected, nothing to save")
        sys.exit(1)
        
    # Generate summary
    summary = collector.validate_collection()
    
    print(f"\nCollection Summary:")
    print(f"Total results: {summary.total_results}")
    print(f"Systems: {', '.join(summary.unique_systems)}")
    print(f"Budgets: {summary.budgets}")
    print(f"Data quality score: {summary.data_quality_score}/100")
    
    if summary.missing_pairs:
        print(f"Warning: {len(summary.missing_pairs)} missing evaluation pairs")
        
    if summary.duplicate_entries:
        print(f"Warning: {len(summary.duplicate_entries)} duplicate entries")
        
    if collector.errors:
        print(f"Warning: {len(collector.errors)} collection errors")
        
    # Save results unless validate-only mode
    if not args.validate_only:
        collector.save_consolidated_results(Path(args.out))
        print(f"\nConsolidated results saved to {args.out}")
        
    # Exit with non-zero code if quality issues
    if summary.data_quality_score < 90:
        sys.exit(1)


if __name__ == "__main__":
    main()
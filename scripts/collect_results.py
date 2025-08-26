#!/usr/bin/env python3
"""
FastPath V5 Results Collection System
====================================

Collects and consolidates evaluation results from all V1-V5 runs for statistical analysis.
Implements the harvest workflow from TODO.md requirements.

Features:
- Consolidates results from baselines and FastPath V5 execution
- Validates data completeness and consistency  
- Prepares data for bootstrap CI computation
- Generates comprehensive result summaries
- Exports data in multiple formats (JSON, CSV, JSONL)

Usage:
    python collect_results.py eval/results                    # Collect from directory
    python collect_results.py eval/results --format csv       # Export as CSV
    python collect_results.py eval/results --validate         # Validate only
"""

import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationRecord:
    """Single evaluation record for statistical analysis."""
    variant_id: str
    variant_type: str  # baseline or fastpath
    run_id: str
    timestamp: str
    
    # Core metrics
    total_tokens: int
    selected_chunks: int
    budget_utilization: float
    coverage_score: float
    diversity_score: float
    execution_time: float
    
    # Performance metrics
    latency_ms: float
    memory_usage_mb: float
    throughput_chunks_per_sec: float
    
    # Algorithm-specific metrics
    algorithm_specific: Dict[str, Any]
    
    # Reproducibility info
    random_seed: int
    repo_path: str
    config_hash: str

@dataclass
class ConsolidatedResults:
    """Consolidated evaluation results for all variants."""
    collection_timestamp: str
    total_records: int
    variant_counts: Dict[str, int]
    baseline_records: List[EvaluationRecord]
    fastpath_records: List[EvaluationRecord]
    data_quality: Dict[str, Any]
    summary_statistics: Dict[str, Any]

class FastPathResultsCollector:
    """Collects and consolidates FastPath V5 evaluation results."""
    
    def __init__(self, results_dir: Path):
        """Initialize collector with results directory."""
        self.results_dir = results_dir
        self.consolidated_results: Optional[ConsolidatedResults] = None
        
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")
        
        logger.info(f"Initialized results collector for: {self.results_dir}")
    
    def collect_all_results(self) -> ConsolidatedResults:
        """Collect and consolidate all evaluation results."""
        logger.info("🔄 Collecting evaluation results from all sources...")
        
        # Collect baseline results (V1-V4)
        baseline_records = self._collect_baseline_results()
        
        # Collect FastPath V5 results
        fastpath_records = self._collect_fastpath_results()
        
        # Validate data quality
        data_quality = self._validate_data_quality(baseline_records + fastpath_records)
        
        # Compute summary statistics
        summary_statistics = self._compute_summary_statistics(baseline_records, fastpath_records)
        
        # Create consolidated results
        variant_counts = self._count_variants(baseline_records + fastpath_records)
        
        self.consolidated_results = ConsolidatedResults(
            collection_timestamp=datetime.utcnow().isoformat(),
            total_records=len(baseline_records) + len(fastpath_records),
            variant_counts=variant_counts,
            baseline_records=baseline_records,
            fastpath_records=fastpath_records,
            data_quality=data_quality,
            summary_statistics=summary_statistics
        )
        
        logger.info(f"✅ Collected {self.consolidated_results.total_records} records")
        logger.info(f"   Baselines: {len(baseline_records)}")
        logger.info(f"   FastPath V5: {len(fastpath_records)}")
        
        return self.consolidated_results
    
    def _collect_baseline_results(self) -> List[EvaluationRecord]:
        """Collect V1-V4 baseline results."""
        logger.info("Collecting V1-V4 baseline results...")
        
        records = []
        
        # Look for baseline summary files
        baseline_patterns = [
            "**/baselines_summary_*.json",
            "**/baselines/baselines_summary_*.json",
            "**/V*_results_*.json"
        ]
        
        baseline_files = []
        for pattern in baseline_patterns:
            baseline_files.extend(self.results_dir.glob(pattern))
        
        logger.info(f"Found {len(baseline_files)} baseline result files")
        
        for result_file in baseline_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                # Handle different baseline file formats
                if "baseline_results" in data:
                    # Summary file with multiple baselines
                    for variant_id, result in data["baseline_results"].items():
                        if result and "selection_result" in result:
                            record = self._extract_baseline_record(variant_id, result, result_file)
                            if record:
                                records.append(record)
                
                elif "baseline_id" in data:
                    # Individual baseline result file
                    record = self._extract_baseline_record(data["baseline_id"], data, result_file)
                    if record:
                        records.append(record)
                
            except Exception as e:
                logger.warning(f"Failed to load baseline file {result_file}: {e}")
        
        logger.info(f"Extracted {len(records)} baseline records")
        return records
    
    def _collect_fastpath_results(self) -> List[EvaluationRecord]:
        """Collect FastPath V5 results."""
        logger.info("Collecting FastPath V5 results...")
        
        records = []
        
        # Look for FastPath summary files
        fastpath_patterns = [
            "**/fastpath_v5_summary_*.json",
            "**/fastpath_v5/fastpath_v5_summary_*.json",
            "**/fastpath_*_results_*.json"
        ]
        
        fastpath_files = []
        for pattern in fastpath_patterns:
            fastpath_files.extend(self.results_dir.glob(pattern))
        
        logger.info(f"Found {len(fastpath_files)} FastPath result files")
        
        for result_file in fastpath_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                # Handle different FastPath file formats
                if "fastpath_results" in data:
                    # Summary file with multiple variants
                    for variant, result in data["fastpath_results"].items():
                        if result and "selection_result" in result:
                            variant_id = f"V5_{variant}"
                            record = self._extract_fastpath_record(variant_id, result, result_file)
                            if record:
                                records.append(record)
                
                elif "variant" in data:
                    # Individual FastPath result file
                    variant_id = f"V5_{data['variant']}"
                    record = self._extract_fastpath_record(variant_id, data, result_file)
                    if record:
                        records.append(record)
                
            except Exception as e:
                logger.warning(f"Failed to load FastPath file {result_file}: {e}")
        
        logger.info(f"Extracted {len(records)} FastPath records")
        return records
    
    def _extract_baseline_record(
        self, 
        variant_id: str, 
        result: Dict[str, Any], 
        source_file: Path
    ) -> Optional[EvaluationRecord]:
        """Extract evaluation record from baseline result."""
        try:
            selection = result.get("selection_result", {})
            execution = result.get("execution", {})
            input_data = result.get("input", {})
            
            # Calculate derived metrics
            total_tokens = selection.get("total_tokens", 0)
            execution_time = execution.get("selection_duration_sec", 0)
            selected_chunks = selection.get("selected_chunks", 0)
            
            throughput = selected_chunks / max(0.001, execution_time)
            latency_ms = execution_time * 1000
            
            # Generate run ID from file and variant
            run_id = f"{source_file.stem}_{variant_id}"
            
            # Create config hash
            config_items = [
                str(input_data.get("token_budget", 0)),
                str(input_data.get("random_seed", 0)),
                variant_id
            ]
            config_hash = str(hash(tuple(config_items)))
            
            return EvaluationRecord(
                variant_id=variant_id,
                variant_type="baseline",
                run_id=run_id,
                timestamp=execution.get("timestamp", datetime.utcnow().isoformat()),
                
                # Core metrics
                total_tokens=total_tokens,
                selected_chunks=selected_chunks,
                budget_utilization=selection.get("budget_utilization", 0.0),
                coverage_score=selection.get("coverage_score", 0.0),
                diversity_score=selection.get("diversity_score", 0.0),
                execution_time=execution_time,
                
                # Performance metrics
                latency_ms=latency_ms,
                memory_usage_mb=50.0,  # Baseline estimate
                throughput_chunks_per_sec=throughput,
                
                # Algorithm-specific metrics
                algorithm_specific={
                    "method": variant_id,
                    "iterations": selection.get("iterations", 1),
                    "avg_selection_score": result.get("chunk_analysis", {}).get("avg_selection_score", 0.0),
                    "files_covered": result.get("chunk_analysis", {}).get("files_covered", 0),
                    "performance_metrics": result.get("performance_metrics", {})
                },
                
                # Reproducibility info
                random_seed=input_data.get("random_seed", 42),
                repo_path=input_data.get("repo_path", "unknown"),
                config_hash=config_hash
            )
            
        except Exception as e:
            logger.error(f"Failed to extract baseline record for {variant_id}: {e}")
            return None
    
    def _extract_fastpath_record(
        self, 
        variant_id: str, 
        result: Dict[str, Any], 
        source_file: Path
    ) -> Optional[EvaluationRecord]:
        """Extract evaluation record from FastPath result."""
        try:
            selection = result.get("selection_result", {})
            execution = result.get("execution", {})
            input_data = result.get("input", {})
            performance = result.get("performance_metrics", {})
            
            # Calculate derived metrics
            total_tokens = selection.get("total_tokens", 0)
            execution_time = execution.get("selection_duration_sec", 0)
            selected_chunks = selection.get("selected_chunks", 0)
            
            throughput = performance.get("throughput_chunks_per_sec", 
                                       selected_chunks / max(0.001, execution_time))
            latency_ms = execution_time * 1000
            memory_mb = execution.get("memory_delta_mb", 100.0)
            
            # Generate run ID from file and variant
            run_id = f"{source_file.stem}_{variant_id}"
            
            # Create config hash
            config_items = [
                str(input_data.get("token_budget", 0)),
                str(input_data.get("random_seed", 0)),
                variant_id
            ]
            config_hash = str(hash(tuple(config_items)))
            
            # Algorithm-specific metrics
            algorithm_specific = {
                "variant": result.get("variant", "unknown"),
                "pagerank_analysis": result.get("pagerank_analysis", {}),
                "feature_analysis": result.get("feature_analysis", {}),
                "chunk_analysis": result.get("chunk_analysis", {}),
                "performance_metrics": performance
            }
            
            return EvaluationRecord(
                variant_id=variant_id,
                variant_type="fastpath",
                run_id=run_id,
                timestamp=execution.get("timestamp", datetime.utcnow().isoformat()),
                
                # Core metrics
                total_tokens=total_tokens,
                selected_chunks=selected_chunks,
                budget_utilization=selection.get("budget_utilization", 0.0),
                coverage_score=selection.get("coverage_score", 0.0),
                diversity_score=selection.get("diversity_score", 0.0),
                execution_time=execution_time,
                
                # Performance metrics
                latency_ms=latency_ms,
                memory_usage_mb=memory_mb,
                throughput_chunks_per_sec=throughput,
                
                # Algorithm-specific metrics
                algorithm_specific=algorithm_specific,
                
                # Reproducibility info
                random_seed=input_data.get("random_seed", 42),
                repo_path=input_data.get("repo_path", "unknown"),
                config_hash=config_hash
            )
            
        except Exception as e:
            logger.error(f"Failed to extract FastPath record for {variant_id}: {e}")
            return None
    
    def _validate_data_quality(self, records: List[EvaluationRecord]) -> Dict[str, Any]:
        """Validate data quality and completeness."""
        logger.info("Validating data quality...")
        
        if not records:
            return {
                "overall_quality": "FAILED",
                "issues": ["No records found"],
                "record_count": 0,
                "completeness": 0.0
            }
        
        issues = []
        
        # Check for required variants
        variants_present = set(record.variant_id for record in records)
        required_baselines = {"V1", "V2", "V3", "V4"}
        missing_baselines = required_baselines - variants_present
        
        if missing_baselines:
            issues.append(f"Missing baseline variants: {missing_baselines}")
        
        # Check for FastPath V5 variants
        fastpath_variants = {v for v in variants_present if v.startswith("V5")}
        if not fastpath_variants:
            issues.append("No FastPath V5 variants found")
        
        # Check data consistency
        token_budgets = set(record.total_tokens for record in records if record.total_tokens > 0)
        if len(token_budgets) > 3:  # Allow some variation
            issues.append(f"Inconsistent token budgets: {len(token_budgets)} different values")
        
        # Check for reasonable values
        unreasonable_records = [
            r for r in records 
            if r.execution_time <= 0 or r.total_tokens <= 0 or r.selected_chunks <= 0
        ]
        if unreasonable_records:
            issues.append(f"{len(unreasonable_records)} records have unreasonable values")
        
        # Check reproducibility info
        missing_seeds = [r for r in records if not hasattr(r, 'random_seed') or r.random_seed is None]
        if missing_seeds:
            issues.append(f"{len(missing_seeds)} records missing random seed")
        
        # Calculate completeness score
        completeness_factors = [
            len(missing_baselines) == 0,  # All baselines present
            len(fastpath_variants) > 0,   # FastPath present
            len(token_budgets) <= 3,      # Consistent budgets
            len(unreasonable_records) == 0,  # Reasonable values
            len(missing_seeds) == 0       # Reproducibility info
        ]
        completeness = sum(completeness_factors) / len(completeness_factors)
        
        # Determine overall quality
        if completeness >= 0.8 and len(issues) <= 1:
            overall_quality = "GOOD"
        elif completeness >= 0.6:
            overall_quality = "ACCEPTABLE"
        else:
            overall_quality = "POOR"
        
        quality_report = {
            "overall_quality": overall_quality,
            "completeness_score": completeness,
            "record_count": len(records),
            "variants_found": list(variants_present),
            "baseline_coverage": len(required_baselines - missing_baselines) / len(required_baselines),
            "fastpath_variants_count": len(fastpath_variants),
            "issues": issues,
            "recommendations": self._generate_quality_recommendations(issues)
        }
        
        logger.info(f"Data quality: {overall_quality} (completeness: {completeness:.2f})")
        if issues:
            logger.warning(f"Quality issues found: {len(issues)}")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return quality_report
    
    def _generate_quality_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations for improving data quality."""
        recommendations = []
        
        for issue in issues:
            if "Missing baseline variants" in issue:
                recommendations.append("Run missing baseline variants using run_baselines.sh")
            elif "No FastPath V5 variants" in issue:
                recommendations.append("Run FastPath V5 variants using run_fastpath.sh")
            elif "Inconsistent token budgets" in issue:
                recommendations.append("Use consistent --budget parameter across all runs")
            elif "unreasonable values" in issue:
                recommendations.append("Check for execution failures and re-run problematic variants")
            elif "missing random seed" in issue:
                recommendations.append("Ensure --seed parameter is specified for reproducibility")
        
        if not recommendations:
            recommendations.append("Data quality is acceptable for statistical analysis")
        
        return recommendations
    
    def _compute_summary_statistics(
        self, 
        baseline_records: List[EvaluationRecord], 
        fastpath_records: List[EvaluationRecord]
    ) -> Dict[str, Any]:
        """Compute summary statistics for all variants."""
        logger.info("Computing summary statistics...")
        
        all_records = baseline_records + fastpath_records
        
        if not all_records:
            return {"error": "No records available for statistics"}
        
        # Group by variant
        by_variant = {}
        for record in all_records:
            if record.variant_id not in by_variant:
                by_variant[record.variant_id] = []
            by_variant[record.variant_id].append(record)
        
        # Compute statistics for each variant
        variant_stats = {}
        for variant_id, records in by_variant.items():
            variant_stats[variant_id] = self._compute_variant_statistics(records)
        
        # Overall statistics
        overall_stats = {
            "total_records": len(all_records),
            "unique_variants": len(by_variant),
            "baseline_variants": len([v for v in by_variant.keys() if not v.startswith("V5")]),
            "fastpath_variants": len([v for v in by_variant.keys() if v.startswith("V5")]),
            
            # Performance ranges
            "execution_time_range": {
                "min": min(r.execution_time for r in all_records),
                "max": max(r.execution_time for r in all_records),
                "mean": np.mean([r.execution_time for r in all_records])
            },
            
            "token_utilization_range": {
                "min": min(r.budget_utilization for r in all_records),
                "max": max(r.budget_utilization for r in all_records), 
                "mean": np.mean([r.budget_utilization for r in all_records])
            },
            
            "coverage_score_range": {
                "min": min(r.coverage_score for r in all_records),
                "max": max(r.coverage_score for r in all_records),
                "mean": np.mean([r.coverage_score for r in all_records])
            }
        }
        
        return {
            "overall": overall_stats,
            "by_variant": variant_stats,
            "comparison_ready": len(by_variant) >= 2,
            "statistical_power": self._estimate_statistical_power(by_variant)
        }
    
    def _compute_variant_statistics(self, records: List[EvaluationRecord]) -> Dict[str, Any]:
        """Compute statistics for a single variant."""
        if not records:
            return {"error": "No records"}
        
        # Extract metrics
        execution_times = [r.execution_time for r in records]
        token_counts = [r.total_tokens for r in records]
        coverage_scores = [r.coverage_score for r in records]
        budget_utilizations = [r.budget_utilization for r in records]
        
        return {
            "record_count": len(records),
            "execution_time": {
                "mean": np.mean(execution_times),
                "std": np.std(execution_times),
                "min": min(execution_times),
                "max": max(execution_times),
                "median": np.median(execution_times)
            },
            "token_usage": {
                "mean": np.mean(token_counts),
                "std": np.std(token_counts),
                "min": min(token_counts),
                "max": max(token_counts)
            },
            "coverage_score": {
                "mean": np.mean(coverage_scores),
                "std": np.std(coverage_scores),
                "min": min(coverage_scores),
                "max": max(coverage_scores)
            },
            "budget_utilization": {
                "mean": np.mean(budget_utilizations),
                "std": np.std(budget_utilizations),
                "min": min(budget_utilizations),
                "max": max(budget_utilizations)
            },
            "reproducibility": {
                "unique_seeds": len(set(r.random_seed for r in records)),
                "unique_configs": len(set(r.config_hash for r in records))
            }
        }
    
    def _estimate_statistical_power(self, by_variant: Dict[str, List[EvaluationRecord]]) -> Dict[str, Any]:
        """Estimate statistical power for comparisons."""
        # Simple power estimation based on sample sizes and effect sizes
        baseline_variants = [v for v in by_variant.keys() if not v.startswith("V5")]
        fastpath_variants = [v for v in by_variant.keys() if v.startswith("V5")]
        
        if not baseline_variants or not fastpath_variants:
            return {"adequate": False, "reason": "Missing baseline or FastPath variants"}
        
        # Check sample sizes
        min_sample_size = min(len(records) for records in by_variant.values())
        adequate_sample_size = min_sample_size >= 3  # Minimum for statistical tests
        
        # Estimate effect size from actual data
        if len(baseline_variants) > 0 and len(fastpath_variants) > 0:
            baseline_performance = np.mean([
                np.mean([r.coverage_score for r in by_variant[v]]) 
                for v in baseline_variants
            ])
            fastpath_performance = np.mean([
                np.mean([r.coverage_score for r in by_variant[v]]) 
                for v in fastpath_variants
            ])
            
            effect_size = abs(fastpath_performance - baseline_performance) / max(0.001, baseline_performance)
        else:
            effect_size = 0.0
        
        return {
            "adequate": adequate_sample_size and effect_size > 0.05,
            "min_sample_size": min_sample_size,
            "estimated_effect_size": effect_size,
            "power_analysis": "Adequate" if adequate_sample_size and effect_size > 0.05 else "Insufficient"
        }
    
    def _count_variants(self, records: List[EvaluationRecord]) -> Dict[str, int]:
        """Count records by variant."""
        counts = {}
        for record in records:
            counts[record.variant_id] = counts.get(record.variant_id, 0) + 1
        return counts
    
    def export_results(self, output_file: Path, format_type: str = "json"):
        """Export consolidated results in specified format."""
        if not self.consolidated_results:
            raise ValueError("No results collected yet - run collect_all_results() first")
        
        logger.info(f"Exporting results to {output_file} (format: {format_type})")
        
        if format_type.lower() == "json":
            self._export_json(output_file)
        elif format_type.lower() == "csv":
            self._export_csv(output_file)
        elif format_type.lower() == "jsonl":
            self._export_jsonl(output_file)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        logger.info(f"✅ Results exported to {output_file}")
    
    def _export_json(self, output_file: Path):
        """Export as JSON."""
        with open(output_file, 'w') as f:
            json.dump(asdict(self.consolidated_results), f, indent=2, default=str)
    
    def _export_csv(self, output_file: Path):
        """Export as CSV."""
        # Flatten evaluation records for CSV
        all_records = self.consolidated_results.baseline_records + self.consolidated_results.fastpath_records
        
        rows = []
        for record in all_records:
            row = asdict(record)
            # Flatten algorithm_specific dict
            algo_specific = row.pop('algorithm_specific', {})
            for key, value in algo_specific.items():
                if isinstance(value, (int, float, str, bool)):
                    row[f'algo_{key}'] = value
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
    
    def _export_jsonl(self, output_file: Path):
        """Export as JSONL (one JSON object per line)."""
        all_records = self.consolidated_results.baseline_records + self.consolidated_results.fastpath_records
        
        with open(output_file, 'w') as f:
            for record in all_records:
                json.dump(asdict(record), f, default=str)
                f.write('\n')
    
    def print_summary(self):
        """Print human-readable summary."""
        if not self.consolidated_results:
            logger.error("No results collected yet")
            return
        
        results = self.consolidated_results
        
        print("\n" + "="*80)
        print("📊 FASTPATH V5 RESULTS COLLECTION SUMMARY")
        print("="*80)
        print(f"Collection Time: {results.collection_timestamp}")
        print(f"Total Records: {results.total_records}")
        print(f"Data Quality: {results.data_quality['overall_quality']}")
        
        print(f"\n📋 VARIANT BREAKDOWN")
        print("-" * 50)
        for variant, count in sorted(results.variant_counts.items()):
            variant_type = "📊 Baseline" if not variant.startswith("V5") else "🚀 FastPath"
            print(f"{variant_type:12} {variant:15} {count:3} records")
        
        print(f"\n🔍 DATA QUALITY ASSESSMENT")
        print("-" * 50)
        quality = results.data_quality
        print(f"Overall Quality: {quality['overall_quality']}")
        print(f"Completeness Score: {quality['completeness_score']:.2f}")
        print(f"Baseline Coverage: {quality['baseline_coverage']:.0%}")
        print(f"FastPath Variants: {quality['fastpath_variants_count']}")
        
        if quality['issues']:
            print(f"\n⚠️ QUALITY ISSUES ({len(quality['issues'])})")
            for i, issue in enumerate(quality['issues'], 1):
                print(f"{i}. {issue}")
        
        if quality['recommendations']:
            print(f"\n💡 RECOMMENDATIONS")
            for i, rec in enumerate(quality['recommendations'], 1):
                print(f"{i}. {rec}")
        
        print(f"\n📈 SUMMARY STATISTICS")
        print("-" * 50)
        overall = results.summary_statistics['overall']
        print(f"Baseline Variants: {overall['baseline_variants']}")
        print(f"FastPath Variants: {overall['fastpath_variants']}")
        print(f"Execution Time Range: {overall['execution_time_range']['min']:.2f} - {overall['execution_time_range']['max']:.2f}s")
        print(f"Coverage Score Range: {overall['coverage_score_range']['min']:.3f} - {overall['coverage_score_range']['max']:.3f}")
        
        statistical_power = results.summary_statistics['statistical_power']
        power_status = "✅" if statistical_power['adequate'] else "❌"
        print(f"\nStatistical Power: {power_status} {statistical_power['power_analysis']}")
        
        print("\n" + "="*80)

def main():
    """Main CLI for results collection."""
    parser = argparse.ArgumentParser(
        description="FastPath V5 Results Collection and Consolidation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s results/                          # Collect from results directory
  %(prog)s results/ --format csv             # Export as CSV 
  %(prog)s results/ --output consolidated    # Custom output file
  %(prog)s results/ --validate               # Validate data quality only
  
Output Formats:
  json   - Structured JSON with full metadata (default)
  csv    - Flat CSV for statistical analysis tools
  jsonl  - JSON Lines format (one record per line)
        """
    )
    
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing evaluation results"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: results_dir/consolidated_results.json)"
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "csv", "jsonl"],
        default="json",
        help="Output format (default: json)"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate data quality only (no export)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input
    if not args.results_dir.exists():
        logger.error(f"Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    # Set default output file
    if not args.output:
        suffix = ".json" if args.format == "json" else f".{args.format}"
        args.output = args.results_dir / f"consolidated_results{suffix}"
    
    try:
        # Initialize collector and collect results
        collector = FastPathResultsCollector(args.results_dir)
        results = collector.collect_all_results()
        
        # Print summary
        collector.print_summary()
        
        # Export if not validation-only mode
        if not args.validate:
            collector.export_results(args.output, args.format)
        
        # Determine exit code based on data quality
        quality = results.data_quality['overall_quality']
        if quality == "GOOD":
            logger.info("✅ Data collection successful - ready for statistical analysis")
            sys.exit(0)
        elif quality == "ACCEPTABLE":
            logger.warning("⚠️ Data collection completed with minor issues")
            sys.exit(0)
        else:
            logger.error("❌ Data quality issues detected - review recommendations")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Results collection failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
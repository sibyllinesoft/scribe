#!/usr/bin/env python3
"""
Inter-Rater Reliability Calculation Tools
FastPath V5 Ground-Truth Protocol - ICSE 2025 Submission

This module implements rigorous statistical validation of annotation quality
using Cohen's kappa and advanced inter-rater reliability measures as required
for academic credibility.

Key Features:
- Cohen's kappa (binary and weighted) calculation
- Krippendorff's alpha for multiple annotators
- Bias detection and statistical validation
- Comprehensive reliability reporting
- ICSE-compliant statistical analysis
"""

import json
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import logging
import argparse
from collections import defaultdict, Counter

from sklearn.metrics import cohen_kappa_score, confusion_matrix
import krippendorff
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class AnnotationData:
    """Structured representation of annotation data."""
    annotator_id: str
    file_path: str
    relevance_score: int  # 1-5 scale
    confidence_score: int  # 1-5 scale
    reasoning: str
    annotation_timestamp: str
    task_batch: str
    repository: str


@dataclass
class ReliabilityAnalysis:
    """Comprehensive reliability analysis results."""
    cohens_kappa_binary: float
    cohens_kappa_weighted: float
    krippendorffs_alpha: float
    agreement_percentage: float
    interpretation: str
    quality_gate_passed: bool
    statistical_significance: Dict[str, Any]
    confidence_intervals: Dict[str, Tuple[float, float]]
    bias_analysis: Dict[str, Any]
    annotator_consistency: Dict[str, float]


class InterRaterReliabilityCalculator:
    """
    Calculate comprehensive inter-rater reliability statistics for annotation quality validation.
    
    Implements academic-grade statistical validation with multiple reliability measures
    and comprehensive bias detection as required by ICSE standards.
    """
    
    def __init__(self, min_kappa_threshold: float = 0.70, 
                 confidence_level: float = 0.95,
                 output_dir: Optional[Path] = None):
        """Initialize reliability calculator with academic standards."""
        self.min_kappa_threshold = min_kappa_threshold
        self.confidence_level = confidence_level
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        
        # Setup logging
        self.setup_logging()
        
        # Statistical constants
        self.kappa_interpretation_thresholds = {
            0.81: "almost_perfect",
            0.61: "substantial", 
            0.41: "moderate",
            0.21: "fair",
            0.01: "slight",
            -1.0: "poor"
        }
        
    def setup_logging(self) -> None:
        """Setup logging for reliability analysis audit trail."""
        log_file = self.output_dir / 'reliability_analysis.log'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_annotations(self, annotation_file: Path) -> List[AnnotationData]:
        """Load and validate annotation data from JSON file."""
        self.logger.info(f"Loading annotations from {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            
        annotations = []
        for item in data.get('annotations', []):
            try:
                annotation = AnnotationData(
                    annotator_id=item['annotator_id'],
                    file_path=item['file_path'],
                    relevance_score=item['relevance_score'],
                    confidence_score=item['confidence_score'],
                    reasoning=item['reasoning'],
                    annotation_timestamp=item['annotation_timestamp'],
                    task_batch=item.get('task_batch', 'default'),
                    repository=item.get('repository', 'unknown')
                )
                annotations.append(annotation)
            except KeyError as e:
                self.logger.error(f"Missing field in annotation data: {e}")
                continue
                
        self.logger.info(f"Loaded {len(annotations)} annotations")
        return annotations
        
    def organize_annotations_by_item(self, annotations: List[AnnotationData]) -> Dict[str, Dict[str, AnnotationData]]:
        """Organize annotations by item (file) for reliability calculation."""
        organized = defaultdict(dict)
        
        for annotation in annotations:
            item_key = f"{annotation.repository}::{annotation.file_path}"
            organized[item_key][annotation.annotator_id] = annotation
            
        # Filter to items with multiple annotations
        multi_annotated = {
            item: annotators for item, annotators in organized.items()
            if len(annotators) >= 2
        }
        
        self.logger.info(f"Found {len(multi_annotated)} items with multiple annotations")
        return multi_annotated
        
    def calculate_cohens_kappa(self, annotations_dict: Dict[str, Dict[str, AnnotationData]]) -> Dict[str, float]:
        """Calculate Cohen's kappa for all annotator pairs."""
        annotators = set()
        for item_annotations in annotations_dict.values():
            annotators.update(item_annotations.keys())
        annotators = list(annotators)
        
        if len(annotators) < 2:
            self.logger.error("Need at least 2 annotators for kappa calculation")
            return {}
            
        kappa_results = {}
        
        # Calculate kappa for all pairs
        for i in range(len(annotators)):
            for j in range(i + 1, len(annotators)):
                annotator_a, annotator_b = annotators[i], annotators[j]
                pair_key = f"{annotator_a}_vs_{annotator_b}"
                
                # Extract paired ratings
                ratings_a, ratings_b = [], []
                for item, item_annotations in annotations_dict.items():
                    if annotator_a in item_annotations and annotator_b in item_annotations:
                        ratings_a.append(item_annotations[annotator_a].relevance_score)
                        ratings_b.append(item_annotations[annotator_b].relevance_score)
                
                if len(ratings_a) < 5:  # Minimum sample size
                    self.logger.warning(f"Insufficient paired annotations for {pair_key}: {len(ratings_a)}")
                    continue
                    
                # Binary kappa (relevant vs not relevant, threshold at 3)
                binary_a = [1 if score >= 3 else 0 for score in ratings_a]
                binary_b = [1 if score >= 3 else 0 for score in ratings_b]
                
                kappa_binary = cohen_kappa_score(binary_a, binary_b)
                
                # Weighted kappa for ordinal scale
                kappa_weighted = cohen_kappa_score(ratings_a, ratings_b, weights='linear')
                
                kappa_results[f"{pair_key}_binary"] = kappa_binary
                kappa_results[f"{pair_key}_weighted"] = kappa_weighted
                
                self.logger.info(f"Cohen's κ for {pair_key}: binary={kappa_binary:.3f}, weighted={kappa_weighted:.3f}")
                
        return kappa_results
        
    def calculate_krippendorffs_alpha(self, annotations_dict: Dict[str, Dict[str, AnnotationData]]) -> float:
        """Calculate Krippendorff's alpha for multiple annotators."""
        
        # Prepare data matrix: annotators x items
        annotators = set()
        items = list(annotations_dict.keys())
        
        for item_annotations in annotations_dict.values():
            annotators.update(item_annotations.keys())
        annotators = sorted(list(annotators))
        
        # Create reliability data matrix
        reliability_data = []
        for annotator in annotators:
            annotator_ratings = []
            for item in items:
                if annotator in annotations_dict[item]:
                    rating = annotations_dict[item][annotator].relevance_score
                else:
                    rating = np.nan  # Missing value
                annotator_ratings.append(rating)
            reliability_data.append(annotator_ratings)
            
        # Calculate Krippendorff's alpha
        try:
            alpha = krippendorff.alpha(reliability_data, level_of_measurement='ordinal')
            self.logger.info(f"Krippendorff's α: {alpha:.3f}")
            return alpha
        except Exception as e:
            self.logger.error(f"Failed to calculate Krippendorff's alpha: {e}")
            return 0.0
            
    def calculate_agreement_percentage(self, annotations_dict: Dict[str, Dict[str, AnnotationData]], 
                                     tolerance: int = 1) -> float:
        """Calculate percentage agreement within tolerance."""
        total_comparisons = 0
        agreements = 0
        
        for item, item_annotations in annotations_dict.items():
            if len(item_annotations) < 2:
                continue
                
            # Get all ratings for this item
            ratings = [ann.relevance_score for ann in item_annotations.values()]
            
            # Count pairwise agreements within tolerance
            for i in range(len(ratings)):
                for j in range(i + 1, len(ratings)):
                    total_comparisons += 1
                    if abs(ratings[i] - ratings[j]) <= tolerance:
                        agreements += 1
                        
        if total_comparisons == 0:
            return 0.0
            
        agreement_percentage = (agreements / total_comparisons) * 100
        self.logger.info(f"Agreement percentage (tolerance={tolerance}): {agreement_percentage:.1f}%")
        return agreement_percentage
        
    def interpret_kappa(self, kappa: float) -> str:
        """Interpret kappa value according to Landis & Koch (1977)."""
        for threshold, interpretation in self.kappa_interpretation_thresholds.items():
            if kappa >= threshold:
                return interpretation
        return "poor"
        
    def calculate_confidence_intervals(self, annotations_dict: Dict[str, Dict[str, AnnotationData]],
                                     kappa_values: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for kappa values."""
        confidence_intervals = {}
        
        # Simplified CI calculation - in production would use bootstrap
        for kappa_name, kappa_value in kappa_values.items():
            if 'binary' in kappa_name:
                # Approximate standard error for binary kappa
                n_items = len([item for item in annotations_dict.keys()
                             if len(annotations_dict[item]) >= 2])
                se = np.sqrt((kappa_value * (1 - kappa_value)) / n_items) if n_items > 0 else 0.1
                
                # 95% confidence interval
                z_score = 1.96  # 95% CI
                ci_lower = max(-1.0, kappa_value - z_score * se)
                ci_upper = min(1.0, kappa_value + z_score * se)
                
                confidence_intervals[kappa_name] = (ci_lower, ci_upper)
                
        return confidence_intervals
        
    def detect_systematic_bias(self, annotations: List[AnnotationData]) -> Dict[str, Any]:
        """Detect systematic biases in annotation patterns."""
        bias_analysis = {
            'annotator_bias': {},
            'file_type_bias': {},
            'repository_bias': {},
            'temporal_bias': {},
            'confidence_alignment_bias': {}
        }
        
        # Annotator-specific bias
        annotator_stats = defaultdict(list)
        for annotation in annotations:
            annotator_stats[annotation.annotator_id].append(annotation.relevance_score)
            
        for annotator_id, scores in annotator_stats.items():
            bias_analysis['annotator_bias'][annotator_id] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'score_distribution': dict(Counter(scores))
            }
            
        # File type bias
        file_type_stats = defaultdict(list)
        for annotation in annotations:
            file_ext = Path(annotation.file_path).suffix.lower()
            file_type_stats[file_ext or 'no_extension'].append(annotation.relevance_score)
            
        for file_type, scores in file_type_stats.items():
            if len(scores) >= 5:  # Minimum sample size
                bias_analysis['file_type_bias'][file_type] = {
                    'mean_score': np.mean(scores),
                    'count': len(scores)
                }
                
        # Confidence-score alignment bias
        confidence_relevance_pairs = [(ann.confidence_score, ann.relevance_score) for ann in annotations]
        if confidence_relevance_pairs:
            confidence_scores, relevance_scores = zip(*confidence_relevance_pairs)
            correlation = np.corrcoef(confidence_scores, relevance_scores)[0, 1]
            bias_analysis['confidence_alignment_bias'] = {
                'correlation': correlation,
                'alignment_quality': 'good' if correlation > 0.5 else 'poor'
            }
            
        return bias_analysis
        
    def calculate_annotator_consistency(self, annotations_dict: Dict[str, Dict[str, AnnotationData]]) -> Dict[str, float]:
        """Calculate individual annotator consistency metrics."""
        annotator_consistency = {}
        
        # Get annotator-specific data
        annotator_data = defaultdict(list)
        for item, item_annotations in annotations_dict.items():
            for annotator_id, annotation in item_annotations.items():
                annotator_data[annotator_id].append({
                    'relevance': annotation.relevance_score,
                    'confidence': annotation.confidence_score,
                    'reasoning_length': len(annotation.reasoning)
                })
                
        for annotator_id, data in annotator_data.items():
            if len(data) < 5:  # Minimum sample size
                continue
                
            relevance_scores = [d['relevance'] for d in data]
            confidence_scores = [d['confidence'] for d in data]
            reasoning_lengths = [d['reasoning_length'] for d in data]
            
            # Consistency metrics
            consistency_score = 1.0 - (np.std(relevance_scores) / 5.0)  # Normalized by scale
            confidence_alignment = np.corrcoef(relevance_scores, confidence_scores)[0, 1]
            reasoning_consistency = 1.0 - (np.std(reasoning_lengths) / np.mean(reasoning_lengths))
            
            # Composite consistency score
            composite_consistency = np.mean([
                consistency_score, 
                (confidence_alignment + 1) / 2,  # Normalize from [-1,1] to [0,1]
                max(0, reasoning_consistency)
            ])
            
            annotator_consistency[annotator_id] = composite_consistency
            
        return annotator_consistency
        
    def perform_statistical_significance_tests(self, kappa_values: Dict[str, float],
                                             n_items: int) -> Dict[str, Any]:
        """Perform statistical significance tests for reliability measures."""
        significance_tests = {}
        
        for kappa_name, kappa_value in kappa_values.items():
            if 'binary' in kappa_name:
                # Test against null hypothesis (κ = 0)
                se = np.sqrt((kappa_value * (1 - kappa_value)) / n_items) if n_items > 0 else 0.1
                z_statistic = kappa_value / se if se > 0 else 0
                p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
                
                significance_tests[kappa_name] = {
                    'z_statistic': z_statistic,
                    'p_value': p_value,
                    'significant_at_05': p_value < 0.05,
                    'effect_size': self.interpret_kappa(kappa_value)
                }
                
        return significance_tests
        
    def generate_reliability_report(self, annotations: List[AnnotationData]) -> ReliabilityAnalysis:
        """Generate comprehensive reliability analysis report."""
        self.logger.info("Generating comprehensive reliability analysis")
        
        # Organize annotations
        annotations_dict = self.organize_annotations_by_item(annotations)
        n_items = len(annotations_dict)
        
        if n_items < 10:
            self.logger.warning(f"Low number of multiply-annotated items: {n_items}")
            
        # Calculate reliability measures
        kappa_values = self.calculate_cohens_kappa(annotations_dict)
        krippendorffs_alpha = self.calculate_krippendorffs_alpha(annotations_dict)
        agreement_percentage = self.calculate_agreement_percentage(annotations_dict)
        
        # Get primary kappa values (average of binary kappa scores)
        binary_kappas = [v for k, v in kappa_values.items() if 'binary' in k]
        weighted_kappas = [v for k, v in kappa_values.items() if 'weighted' in k]
        
        avg_binary_kappa = np.mean(binary_kappas) if binary_kappas else 0.0
        avg_weighted_kappa = np.mean(weighted_kappas) if weighted_kappas else 0.0
        
        # Interpretation and quality gate
        interpretation = self.interpret_kappa(avg_binary_kappa)
        quality_gate_passed = avg_binary_kappa >= self.min_kappa_threshold
        
        # Statistical validation
        confidence_intervals = self.calculate_confidence_intervals(annotations_dict, kappa_values)
        significance_tests = self.perform_statistical_significance_tests(kappa_values, n_items)
        
        # Bias analysis
        bias_analysis = self.detect_systematic_bias(annotations)
        
        # Annotator consistency
        annotator_consistency = self.calculate_annotator_consistency(annotations_dict)
        
        # Create comprehensive analysis
        analysis = ReliabilityAnalysis(
            cohens_kappa_binary=avg_binary_kappa,
            cohens_kappa_weighted=avg_weighted_kappa,
            krippendorffs_alpha=krippendorffs_alpha,
            agreement_percentage=agreement_percentage,
            interpretation=interpretation,
            quality_gate_passed=quality_gate_passed,
            statistical_significance=significance_tests,
            confidence_intervals=confidence_intervals,
            bias_analysis=bias_analysis,
            annotator_consistency=annotator_consistency
        )
        
        return analysis
        
    def create_visualization_report(self, analysis: ReliabilityAnalysis,
                                  annotations: List[AnnotationData]) -> Path:
        """Create comprehensive visualization report."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Inter-Rater Reliability Analysis Report', fontsize=16, fontweight='bold')
        
        # 1. Kappa values comparison
        ax1 = axes[0, 0]
        kappa_types = ['Binary κ', 'Weighted κ', "Krippendorff's α"]
        kappa_values = [
            analysis.cohens_kappa_binary,
            analysis.cohens_kappa_weighted, 
            analysis.krippendorffs_alpha
        ]
        
        bars = ax1.bar(kappa_types, kappa_values, 
                      color=['#2E86AB', '#A23B72', '#F18F01'])
        ax1.axhline(y=self.min_kappa_threshold, color='red', linestyle='--', 
                   label=f'Threshold ({self.min_kappa_threshold})')
        ax1.set_ylabel('Reliability Coefficient')
        ax1.set_title('Reliability Measures')
        ax1.legend()
        ax1.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for bar, value in zip(bars, kappa_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Score distribution by annotator
        ax2 = axes[0, 1]
        annotator_scores = defaultdict(list)
        for ann in annotations:
            annotator_scores[ann.annotator_id].append(ann.relevance_score)
            
        for i, (annotator, scores) in enumerate(annotator_scores.items()):
            ax2.hist(scores, bins=5, alpha=0.6, label=annotator, 
                    range=(0.5, 5.5), density=True)
        ax2.set_xlabel('Relevance Score')
        ax2.set_ylabel('Density')
        ax2.set_title('Score Distribution by Annotator')
        ax2.legend()
        
        # 3. Confidence vs Relevance alignment
        ax3 = axes[0, 2]
        confidence_scores = [ann.confidence_score for ann in annotations]
        relevance_scores = [ann.relevance_score for ann in annotations]
        ax3.scatter(confidence_scores, relevance_scores, alpha=0.6)
        ax3.set_xlabel('Confidence Score')
        ax3.set_ylabel('Relevance Score')
        ax3.set_title('Confidence-Relevance Alignment')
        
        # Add correlation line
        z = np.polyfit(confidence_scores, relevance_scores, 1)
        p = np.poly1d(z)
        ax3.plot(confidence_scores, p(confidence_scores), "r--", alpha=0.8)
        correlation = np.corrcoef(confidence_scores, relevance_scores)[0, 1]
        ax3.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax3.transAxes,
                fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Annotator consistency scores
        ax4 = axes[1, 0]
        if analysis.annotator_consistency:
            annotators = list(analysis.annotator_consistency.keys())
            consistency_scores = list(analysis.annotator_consistency.values())
            
            bars = ax4.bar(annotators, consistency_scores, color='#52B788')
            ax4.set_ylabel('Consistency Score')
            ax4.set_title('Annotator Consistency')
            ax4.set_ylim(0, 1.0)
            
            # Add value labels
            for bar, value in zip(bars, consistency_scores):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor consistency analysis',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Annotator Consistency')
            
        # 5. Agreement matrix (if multiple annotators)
        ax5 = axes[1, 1]
        annotators = list(set(ann.annotator_id for ann in annotations))
        if len(annotators) >= 2:
            # Create agreement matrix
            agreement_matrix = np.zeros((len(annotators), len(annotators)))
            
            # This is simplified - full implementation would calculate pairwise agreements
            for i in range(len(annotators)):
                for j in range(len(annotators)):
                    if i == j:
                        agreement_matrix[i, j] = 1.0
                    else:
                        # Use binary kappa as proxy for agreement
                        agreement_matrix[i, j] = analysis.cohens_kappa_binary
                        
            im = ax5.imshow(agreement_matrix, cmap='RdYlBu', vmin=0, vmax=1)
            ax5.set_xticks(range(len(annotators)))
            ax5.set_yticks(range(len(annotators)))
            ax5.set_xticklabels(annotators, rotation=45)
            ax5.set_yticklabels(annotators)
            ax5.set_title('Pairwise Agreement Matrix')
            plt.colorbar(im, ax=ax5)
        else:
            ax5.text(0.5, 0.5, 'Need multiple annotators\nfor agreement matrix',
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Pairwise Agreement Matrix')
        
        # 6. Quality gate status
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Quality gate summary
        gate_status = "✅ PASSED" if analysis.quality_gate_passed else "❌ FAILED"
        gate_color = 'green' if analysis.quality_gate_passed else 'red'
        
        summary_text = f"""
        QUALITY GATE: {gate_status}
        
        Cohen's κ (binary): {analysis.cohens_kappa_binary:.3f}
        Threshold: {self.min_kappa_threshold}
        Interpretation: {analysis.interpretation.upper()}
        
        Agreement %: {analysis.agreement_percentage:.1f}%
        Krippendorff's α: {analysis.krippendorffs_alpha:.3f}
        
        Annotators: {len(set(ann.annotator_id for ann in annotations))}
        Items annotated: {len(set(f"{ann.repository}::{ann.file_path}" for ann in annotations))}
        """
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor=gate_color, alpha=0.1))
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.output_dir / 'reliability_analysis_report.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return viz_path
        
    def save_analysis_results(self, analysis: ReliabilityAnalysis, 
                            annotations: List[AnnotationData],
                            output_filename: str = 'reliability_analysis.json') -> Path:
        """Save comprehensive reliability analysis results."""
        
        # Prepare serializable analysis data
        analysis_data = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'calculator_version': '1.0.0',
                'methodology': 'ground_truth_protocol_v1.0',
                'min_kappa_threshold': self.min_kappa_threshold,
                'confidence_level': self.confidence_level
            },
            'reliability_results': asdict(analysis),
            'dataset_summary': {
                'total_annotations': len(annotations),
                'unique_annotators': len(set(ann.annotator_id for ann in annotations)),
                'unique_items': len(set(f"{ann.repository}::{ann.file_path}" for ann in annotations)),
                'repositories': list(set(ann.repository for ann in annotations)),
                'task_batches': list(set(ann.task_batch for ann in annotations))
            },
            'quality_assessment': {
                'icse_standards_compliance': {
                    'kappa_threshold_met': analysis.quality_gate_passed,
                    'multiple_annotators': len(set(ann.annotator_id for ann in annotations)) >= 2,
                    'sufficient_sample_size': len(annotations) >= 50,
                    'bias_analysis_complete': bool(analysis.bias_analysis),
                    'statistical_significance_tested': bool(analysis.statistical_significance)
                },
                'recommendation': (
                    "ACCEPT - Dataset meets academic standards for publication"
                    if analysis.quality_gate_passed 
                    else "REJECT - Additional annotation rounds required"
                )
            }
        }
        
        # Save to JSON
        output_path = self.output_dir / output_filename
        with open(output_path, 'w') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False, default=str)
            
        self.logger.info(f"Reliability analysis saved to {output_path}")
        return output_path


def main():
    """Main execution function for inter-rater reliability calculation."""
    parser = argparse.ArgumentParser(
        description='Calculate inter-rater reliability for ground-truth annotations'
    )
    parser.add_argument('--annotation-file', required=True, type=Path,
                       help='JSON file containing annotation data')
    parser.add_argument('--output-dir', required=True, type=Path,
                       help='Output directory for analysis results')
    parser.add_argument('--min-kappa', type=float, default=0.70,
                       help='Minimum acceptable Cohen\'s kappa (default: 0.70)')
    parser.add_argument('--confidence-level', type=float, default=0.95,
                       help='Confidence level for statistical tests (default: 0.95)')
    parser.add_argument('--create-visualizations', action='store_true',
                       help='Generate visualization report')
    
    args = parser.parse_args()
    
    # Initialize calculator
    calculator = InterRaterReliabilityCalculator(
        min_kappa_threshold=args.min_kappa,
        confidence_level=args.confidence_level,
        output_dir=args.output_dir
    )
    
    try:
        # Load annotation data
        annotations = calculator.load_annotations(args.annotation_file)
        
        if len(annotations) < 10:
            print(f"⚠️  Warning: Low annotation count ({len(annotations)}). Results may be unreliable.")
            
        # Perform reliability analysis
        analysis = calculator.generate_reliability_report(annotations)
        
        # Save results
        results_path = calculator.save_analysis_results(analysis, annotations)
        
        # Create visualizations if requested
        viz_path = None
        if args.create_visualizations:
            viz_path = calculator.create_visualization_report(analysis, annotations)
            
        # Print summary
        print("\n" + "="*60)
        print("INTER-RATER RELIABILITY ANALYSIS COMPLETE")
        print("="*60)
        print(f"📊 Cohen's κ (binary): {analysis.cohens_kappa_binary:.3f}")
        print(f"📊 Cohen's κ (weighted): {analysis.cohens_kappa_weighted:.3f}")
        print(f"📊 Krippendorff's α: {analysis.krippendorffs_alpha:.3f}")
        print(f"📊 Agreement percentage: {analysis.agreement_percentage:.1f}%")
        print(f"📊 Interpretation: {analysis.interpretation.upper()}")
        print(f"🎯 Quality gate: {'✅ PASSED' if analysis.quality_gate_passed else '❌ FAILED'}")
        print(f"📁 Results saved to: {results_path}")
        if viz_path:
            print(f"📈 Visualizations saved to: {viz_path}")
        print("="*60)
        
        # Exit code for automation
        exit_code = 0 if analysis.quality_gate_passed else 1
        
    except Exception as e:
        calculator.logger.error(f"Reliability analysis failed: {e}")
        print(f"❌ Analysis failed: {e}")
        exit_code = 2
        
    exit(exit_code)


if __name__ == "__main__":
    main()
"""
Quality Assurance and Inter-Annotator Agreement for QA Datasets

Implements comprehensive quality assurance protocols including inter-annotator
agreement measurement (κ≥0.6 required by TODO.md), consistency checking,
and automated quality metrics to ensure dataset reliability.

Key Features:
- Inter-annotator agreement calculation (Cohen's κ, Fleiss' κ)
- Quality control metrics and thresholds
- Annotation consistency validation
- Statistical significance testing
- Automated quality improvement recommendations
"""

import json
import random
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np
from scipy import stats
import logging

from .schema import QuestionItem, DifficultyLevel, QuestionCategory

logger = logging.getLogger(__name__)


@dataclass
class AnnotatorRating:
    """Individual annotator's rating for a question."""
    annotator_id: str
    question_id: str
    score: float  # 0-1 scale
    confidence: float  # Annotator's confidence in their rating
    notes: str = ""
    timestamp: str = ""


@dataclass
class InterAnnotatorAgreement:
    """Inter-annotator agreement statistics."""
    cohens_kappa: float
    fleiss_kappa: Optional[float] = None  # For >2 annotators
    agreement_percentage: float = 0.0
    correlation_coefficient: float = 0.0
    sample_size: int = 0
    annotator_count: int = 0
    meets_threshold: bool = False  # κ≥0.6 requirement
    confidence_interval: Tuple[float, float] = (0.0, 0.0)


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for dataset."""
    total_questions: int
    validation_pass_rate: float
    mean_confidence_score: float
    difficulty_distribution: Dict[str, float]
    category_distribution: Dict[str, float]
    inter_annotator_agreement: InterAnnotatorAgreement
    consistency_score: float
    completeness_score: float
    quality_issues: List[str]
    recommendations: List[str]


def cohens_kappa(ratings1: List[float], ratings2: List[float], weights: str = 'linear') -> float:
    """
    Calculate Cohen's kappa for two raters.
    
    Args:
        ratings1: Ratings from first annotator
        ratings2: Ratings from second annotator  
        weights: Weighting scheme ('linear' or 'quadratic')
        
    Returns:
        Cohen's kappa coefficient (-1 to 1)
    """
    if len(ratings1) != len(ratings2):
        raise ValueError("Rating lists must have equal length")
    
    if not ratings1:
        return 0.0
    
    # Convert to discrete categories (0-4 scale for calculation)
    cats1 = [min(4, int(r * 4)) for r in ratings1]
    cats2 = [min(4, int(r * 4)) for r in ratings2]
    
    n = len(cats1)
    categories = list(range(5))  # 0-4
    
    # Create confusion matrix
    confusion_matrix = np.zeros((len(categories), len(categories)))
    for c1, c2 in zip(cats1, cats2):
        confusion_matrix[c1, c2] += 1
    
    # Calculate observed agreement
    observed_agreement = np.trace(confusion_matrix) / n
    
    # Calculate expected agreement
    marginal1 = np.sum(confusion_matrix, axis=1) / n
    marginal2 = np.sum(confusion_matrix, axis=0) / n
    expected_agreement = np.sum(marginal1 * marginal2)
    
    # Calculate Cohen's kappa
    if expected_agreement == 1.0:
        return 1.0  # Perfect expected agreement
    
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    
    # Apply weighting if specified
    if weights == 'quadratic':
        # Apply quadratic weighting (higher penalty for larger disagreements)
        weighted_confusion = np.zeros_like(confusion_matrix)
        for i in range(len(categories)):
            for j in range(len(categories)):
                weight = 1 - ((i - j) ** 2) / ((len(categories) - 1) ** 2)
                weighted_confusion[i, j] = confusion_matrix[i, j] * weight
        
        weighted_observed = np.sum(weighted_confusion) / n
        weighted_expected = 0.0
        for i in range(len(categories)):
            for j in range(len(categories)):
                weight = 1 - ((i - j) ** 2) / ((len(categories) - 1) ** 2)
                weighted_expected += marginal1[i] * marginal2[j] * weight
        
        if weighted_expected != 1.0:
            kappa = (weighted_observed - weighted_expected) / (1 - weighted_expected)
    
    return kappa


def fleiss_kappa(ratings: List[List[float]]) -> float:
    """
    Calculate Fleiss' kappa for multiple raters.
    
    Args:
        ratings: List of rating lists, one per annotator
        
    Returns:
        Fleiss' kappa coefficient
    """
    if len(ratings) < 2:
        return 0.0
    
    n_annotators = len(ratings)
    n_items = len(ratings[0])
    
    # Check all rating lists have same length
    if not all(len(rating_list) == n_items for rating_list in ratings):
        raise ValueError("All rating lists must have equal length")
    
    # Convert to discrete categories
    categories = 5  # 0-4 scale
    category_ratings = []
    for rating_list in ratings:
        cats = [min(4, int(r * 4)) for r in rating_list]
        category_ratings.append(cats)
    
    # Create rating matrix (items x categories)
    rating_matrix = np.zeros((n_items, categories))
    for item_idx in range(n_items):
        for annotator_idx in range(n_annotators):
            category = category_ratings[annotator_idx][item_idx]
            rating_matrix[item_idx, category] += 1
    
    # Calculate observed agreement
    P = np.zeros(n_items)
    for i in range(n_items):
        P[i] = (np.sum(rating_matrix[i] ** 2) - n_annotators) / (n_annotators * (n_annotators - 1))
    
    P_bar = np.mean(P)
    
    # Calculate expected agreement
    p_j = np.sum(rating_matrix, axis=0) / (n_items * n_annotators)
    P_e_bar = np.sum(p_j ** 2)
    
    # Calculate Fleiss' kappa
    if P_e_bar == 1.0:
        return 1.0
    
    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
    return kappa


def calculate_confidence_interval(kappa: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for kappa statistic."""
    if n <= 1:
        return (kappa, kappa)
    
    # Simplified calculation - for production use, implement proper SE calculation
    z_score = stats.norm.ppf((1 + confidence) / 2)
    se_approx = np.sqrt((1 - kappa ** 2) / n)  # Rough approximation
    
    lower = kappa - z_score * se_approx
    upper = kappa + z_score * se_approx
    
    # Clamp to valid kappa range
    lower = max(-1.0, lower)
    upper = min(1.0, upper)
    
    return (lower, upper)


class QualityAssurance:
    """Quality assurance system for QA datasets."""
    
    def __init__(self, kappa_threshold: float = 0.6):
        """
        Initialize QA system.
        
        Args:
            kappa_threshold: Minimum acceptable kappa (TODO.md requires ≥0.6)
        """
        self.kappa_threshold = kappa_threshold
        self.quality_standards = self._define_quality_standards()
    
    def _define_quality_standards(self) -> Dict[str, Any]:
        """Define quality standards and thresholds."""
        return {
            'inter_annotator_agreement': {
                'min_kappa': self.kappa_threshold,
                'min_sample_size': 50,  # Minimum for reliable κ calculation
                'min_annotators': 2
            },
            'validation_requirements': {
                'min_pass_rate': 0.85,  # 85% questions must pass validation
                'min_confidence': 0.7   # Average confidence ≥0.7
            },
            'distribution_requirements': {
                'difficulty': {
                    'easy': (0.15, 0.25),    # 15-25% easy questions
                    'medium': (0.40, 0.60),   # 40-60% medium questions  
                    'hard': (0.25, 0.35)     # 25-35% hard questions
                },
                'min_questions_per_category': 5,
                'max_category_imbalance': 0.4  # No category >40% of total
            }
        }
    
    def measure_inter_annotator_agreement(
        self,
        ratings: List[AnnotatorRating],
        sample_questions: Optional[List[str]] = None
    ) -> InterAnnotatorAgreement:
        """
        Measure inter-annotator agreement on sample of questions.
        
        Args:
            ratings: List of annotator ratings
            sample_questions: Specific questions to analyze (None = all)
            
        Returns:
            InterAnnotatorAgreement statistics
        """
        if not ratings:
            return InterAnnotatorAgreement(
                cohens_kappa=0.0,
                meets_threshold=False
            )
        
        # Filter by sample questions if specified
        if sample_questions:
            ratings = [r for r in ratings if r.question_id in sample_questions]
        
        # Group ratings by question and annotator
        question_ratings = defaultdict(dict)
        annotators = set()
        
        for rating in ratings:
            question_ratings[rating.question_id][rating.annotator_id] = rating.score
            annotators.add(rating.annotator_id)
        
        annotators = sorted(list(annotators))
        
        if len(annotators) < 2:
            logger.warning("Need at least 2 annotators for agreement calculation")
            return InterAnnotatorAgreement(
                cohens_kappa=0.0,
                annotator_count=len(annotators),
                meets_threshold=False
            )
        
        # Filter questions with ratings from all annotators
        complete_questions = []
        for qid, q_ratings in question_ratings.items():
            if len(q_ratings) == len(annotators):
                complete_questions.append(qid)
        
        if not complete_questions:
            logger.warning("No questions with ratings from all annotators")
            return InterAnnotatorAgreement(
                cohens_kappa=0.0,
                annotator_count=len(annotators),
                sample_size=0,
                meets_threshold=False
            )
        
        # Prepare rating matrices
        rating_matrix = []
        for annotator in annotators:
            annotator_ratings = []
            for qid in complete_questions:
                annotator_ratings.append(question_ratings[qid][annotator])
            rating_matrix.append(annotator_ratings)
        
        # Calculate agreement statistics
        if len(annotators) == 2:
            # Cohen's kappa for 2 annotators
            kappa = cohens_kappa(rating_matrix[0], rating_matrix[1])
            fleiss_k = None
        else:
            # Fleiss' kappa for >2 annotators
            kappa = fleiss_kappa(rating_matrix)
            fleiss_k = kappa
        
        # Calculate other metrics
        agreement_percentage = self._calculate_agreement_percentage(rating_matrix)
        correlation = self._calculate_correlation(rating_matrix)
        confidence_interval = calculate_confidence_interval(kappa, len(complete_questions))
        
        return InterAnnotatorAgreement(
            cohens_kappa=kappa,
            fleiss_kappa=fleiss_k,
            agreement_percentage=agreement_percentage,
            correlation_coefficient=correlation,
            sample_size=len(complete_questions),
            annotator_count=len(annotators),
            meets_threshold=kappa >= self.kappa_threshold,
            confidence_interval=confidence_interval
        )
    
    def _calculate_agreement_percentage(self, rating_matrix: List[List[float]]) -> float:
        """Calculate percentage exact agreement."""
        if not rating_matrix or len(rating_matrix) < 2:
            return 0.0
        
        n_items = len(rating_matrix[0])
        exact_agreements = 0
        
        for i in range(n_items):
            item_ratings = [ratings[i] for ratings in rating_matrix]
            # Consider agreement within 0.1 threshold
            if max(item_ratings) - min(item_ratings) <= 0.1:
                exact_agreements += 1
        
        return exact_agreements / n_items if n_items > 0 else 0.0
    
    def _calculate_correlation(self, rating_matrix: List[List[float]]) -> float:
        """Calculate average pairwise correlation."""
        if len(rating_matrix) < 2:
            return 0.0
        
        correlations = []
        for i in range(len(rating_matrix)):
            for j in range(i + 1, len(rating_matrix)):
                corr, _ = stats.pearsonr(rating_matrix[i], rating_matrix[j])
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def evaluate_dataset_quality(
        self,
        questions: List[QuestionItem],
        annotation_ratings: List[AnnotatorRating] = None
    ) -> QualityMetrics:
        """
        Comprehensive quality evaluation of QA dataset.
        
        Args:
            questions: List of questions to evaluate
            annotation_ratings: Optional annotator ratings for agreement calculation
            
        Returns:
            Complete quality metrics report
        """
        issues = []
        recommendations = []
        
        # Basic statistics
        total_questions = len(questions)
        
        if total_questions == 0:
            return QualityMetrics(
                total_questions=0,
                validation_pass_rate=0.0,
                mean_confidence_score=0.0,
                difficulty_distribution={},
                category_distribution={},
                inter_annotator_agreement=InterAnnotatorAgreement(cohens_kappa=0.0, meets_threshold=False),
                consistency_score=0.0,
                completeness_score=0.0,
                quality_issues=["No questions in dataset"],
                recommendations=["Add questions to dataset"]
            )
        
        # Distribution analysis
        difficulty_dist = self._analyze_difficulty_distribution(questions)
        category_dist = self._analyze_category_distribution(questions)
        
        # Validation analysis
        validation_results = self._analyze_validation_status(questions)
        
        # Inter-annotator agreement
        if annotation_ratings:
            iaa = self.measure_inter_annotator_agreement(annotation_ratings)
        else:
            iaa = InterAnnotatorAgreement(cohens_kappa=0.0, meets_threshold=False)
            issues.append("No annotator ratings provided - cannot measure inter-annotator agreement")
        
        # Consistency analysis
        consistency_score = self._calculate_consistency_score(questions)
        
        # Completeness analysis
        completeness_score = self._calculate_completeness_score(questions)
        
        # Quality issues and recommendations
        quality_issues, quality_recommendations = self._identify_quality_issues(
            questions, difficulty_dist, category_dist, validation_results, iaa
        )
        
        issues.extend(quality_issues)
        recommendations.extend(quality_recommendations)
        
        return QualityMetrics(
            total_questions=total_questions,
            validation_pass_rate=validation_results['pass_rate'],
            mean_confidence_score=validation_results['mean_confidence'],
            difficulty_distribution=difficulty_dist,
            category_distribution=category_dist,
            inter_annotator_agreement=iaa,
            consistency_score=consistency_score,
            completeness_score=completeness_score,
            quality_issues=issues,
            recommendations=recommendations
        )
    
    def _analyze_difficulty_distribution(self, questions: List[QuestionItem]) -> Dict[str, float]:
        """Analyze distribution of question difficulties."""
        if not questions:
            return {}
        
        difficulty_counts = Counter(q.difficulty.value for q in questions)
        total = len(questions)
        
        return {
            difficulty: count / total
            for difficulty, count in difficulty_counts.items()
        }
    
    def _analyze_category_distribution(self, questions: List[QuestionItem]) -> Dict[str, float]:
        """Analyze distribution of question categories."""
        if not questions:
            return {}
        
        category_counts = Counter(q.category.value for q in questions)
        total = len(questions)
        
        return {
            category: count / total
            for category, count in category_counts.items()
        }
    
    def _analyze_validation_status(self, questions: List[QuestionItem]) -> Dict[str, float]:
        """Analyze validation status of questions."""
        if not questions:
            return {'pass_rate': 0.0, 'mean_confidence': 0.0}
        
        # Simple validation based on presence of required fields
        valid_questions = 0
        total_confidence = 0.0
        
        for question in questions:
            is_valid = True
            
            # Check required fields
            if not question.question.strip():
                is_valid = False
            if not question.gold or not question.gold.key_concepts:
                is_valid = False
            
            if is_valid:
                valid_questions += 1
            
            total_confidence += question.gold.confidence_score if question.gold else 0.0
        
        return {
            'pass_rate': valid_questions / len(questions),
            'mean_confidence': total_confidence / len(questions)
        }
    
    def _calculate_consistency_score(self, questions: List[QuestionItem]) -> float:
        """Calculate consistency score based on question structure."""
        if not questions:
            return 0.0
        
        consistency_factors = []
        
        # Check for consistent question length
        question_lengths = [len(q.question.split()) for q in questions]
        if question_lengths:
            length_std = np.std(question_lengths)
            length_mean = np.mean(question_lengths)
            length_consistency = max(0, 1 - (length_std / max(1, length_mean)))
            consistency_factors.append(length_consistency)
        
        # Check for consistent concept count
        concept_counts = []
        for q in questions:
            if q.gold and q.gold.key_concepts:
                concept_counts.append(len(q.gold.key_concepts))
        
        if concept_counts:
            concept_std = np.std(concept_counts)
            concept_mean = np.mean(concept_counts)
            concept_consistency = max(0, 1 - (concept_std / max(1, concept_mean)))
            consistency_factors.append(concept_consistency)
        
        # Check for consistent confidence scores
        confidence_scores = []
        for q in questions:
            if q.gold:
                confidence_scores.append(q.gold.confidence_score)
        
        if confidence_scores:
            conf_std = np.std(confidence_scores)
            conf_consistency = max(0, 1 - conf_std)  # Lower std = higher consistency
            consistency_factors.append(conf_consistency)
        
        return np.mean(consistency_factors) if consistency_factors else 0.0
    
    def _calculate_completeness_score(self, questions: List[QuestionItem]) -> float:
        """Calculate completeness score based on required information."""
        if not questions:
            return 0.0
        
        completeness_scores = []
        
        for question in questions:
            score = 0.0
            total_checks = 5
            
            # Check question text
            if question.question and len(question.question.strip()) > 10:
                score += 1
            
            # Check gold answer
            if question.gold:
                # Has key concepts
                if question.gold.key_concepts and len(question.gold.key_concepts) >= 2:
                    score += 1
                
                # Has answer text or rubric
                if question.gold.answer_text or question.gold.evaluation_rubric:
                    score += 1
                
                # Has reasonable confidence
                if question.gold.confidence_score >= 0.5:
                    score += 1
            
            # Check rubric for rubric-based questions
            if question.evaluation_type.value == 'rubric_based':
                if question.rubric and len(question.rubric) > 50:
                    score += 1
            else:
                score += 1  # Not required for non-rubric questions
            
            completeness_scores.append(score / total_checks)
        
        return np.mean(completeness_scores)
    
    def _identify_quality_issues(
        self,
        questions: List[QuestionItem],
        difficulty_dist: Dict[str, float],
        category_dist: Dict[str, float],
        validation_results: Dict[str, float],
        iaa: InterAnnotatorAgreement
    ) -> Tuple[List[str], List[str]]:
        """Identify quality issues and generate recommendations."""
        issues = []
        recommendations = []
        
        standards = self.quality_standards
        
        # Check total question count
        if len(questions) < 300:
            issues.append(f"Insufficient questions: {len(questions)} < 300 required")
            recommendations.append("Generate more questions to meet minimum requirement")
        
        # Check difficulty distribution
        difficulty_reqs = standards['distribution_requirements']['difficulty']
        for difficulty, (min_pct, max_pct) in difficulty_reqs.items():
            actual_pct = difficulty_dist.get(difficulty, 0.0)
            if actual_pct < min_pct:
                issues.append(f"Too few {difficulty} questions: {actual_pct:.1%} < {min_pct:.1%}")
                recommendations.append(f"Add more {difficulty} questions")
            elif actual_pct > max_pct:
                issues.append(f"Too many {difficulty} questions: {actual_pct:.1%} > {max_pct:.1%}")
                recommendations.append(f"Reduce {difficulty} questions or add others")
        
        # Check category balance
        max_category_pct = max(category_dist.values()) if category_dist else 0
        max_imbalance = standards['distribution_requirements']['max_category_imbalance']
        if max_category_pct > max_imbalance:
            issues.append(f"Category imbalance: {max_category_pct:.1%} > {max_imbalance:.1%}")
            recommendations.append("Balance question categories more evenly")
        
        # Check validation rate
        min_pass_rate = standards['validation_requirements']['min_pass_rate']
        if validation_results['pass_rate'] < min_pass_rate:
            issues.append(f"Low validation rate: {validation_results['pass_rate']:.1%} < {min_pass_rate:.1%}")
            recommendations.append("Improve question and gold answer quality")
        
        # Check confidence scores
        min_confidence = standards['validation_requirements']['min_confidence']
        if validation_results['mean_confidence'] < min_confidence:
            issues.append(f"Low confidence: {validation_results['mean_confidence']:.2f} < {min_confidence}")
            recommendations.append("Review and improve gold answer quality")
        
        # Check inter-annotator agreement
        if not iaa.meets_threshold:
            issues.append(f"Low inter-annotator agreement: κ={iaa.cohens_kappa:.3f} < {self.kappa_threshold}")
            recommendations.append("Improve annotation guidelines and training")
        
        if iaa.sample_size < standards['inter_annotator_agreement']['min_sample_size']:
            issues.append(f"Insufficient annotation sample: {iaa.sample_size} < {standards['inter_annotator_agreement']['min_sample_size']}")
            recommendations.append("Collect more annotator ratings for reliable statistics")
        
        return issues, recommendations
    
    def generate_annotation_sample(
        self,
        questions: List[QuestionItem],
        sample_size: int = 50,
        stratify: bool = True
    ) -> List[QuestionItem]:
        """
        Generate stratified sample for annotation agreement testing.
        
        Args:
            questions: Full question list
            sample_size: Target sample size (TODO.md requires ≥50)
            stratify: Whether to stratify by difficulty and category
            
        Returns:
            Sample of questions for annotation
        """
        if len(questions) <= sample_size:
            return questions.copy()
        
        if not stratify:
            return random.sample(questions, sample_size)
        
        # Stratified sampling by difficulty and category
        sample = []
        
        # Group by difficulty
        difficulty_groups = defaultdict(list)
        for q in questions:
            difficulty_groups[q.difficulty].append(q)
        
        # Calculate target sizes per difficulty
        difficulty_targets = {
            DifficultyLevel.EASY: int(sample_size * 0.2),
            DifficultyLevel.MEDIUM: int(sample_size * 0.5),
            DifficultyLevel.HARD: int(sample_size * 0.3)
        }
        
        # Adjust for actual availability
        total_target = sum(difficulty_targets.values())
        if total_target < sample_size:
            # Distribute remaining to medium
            difficulty_targets[DifficultyLevel.MEDIUM] += sample_size - total_target
        
        # Sample from each difficulty group
        for difficulty, target_count in difficulty_targets.items():
            available = difficulty_groups[difficulty]
            if available:
                sample_count = min(target_count, len(available))
                sample.extend(random.sample(available, sample_count))
        
        # Fill remaining spots if needed
        remaining = sample_size - len(sample)
        if remaining > 0:
            used_ids = {q.qid for q in sample}
            unused_questions = [q for q in questions if q.qid not in used_ids]
            if unused_questions:
                additional_sample = random.sample(unused_questions, min(remaining, len(unused_questions)))
                sample.extend(additional_sample)
        
        return sample
    
    def export_quality_report(self, metrics: QualityMetrics, output_file: Path) -> None:
        """Export comprehensive quality report."""
        report = {
            'dataset_summary': {
                'total_questions': metrics.total_questions,
                'validation_pass_rate': metrics.validation_pass_rate,
                'mean_confidence_score': metrics.mean_confidence_score,
                'consistency_score': metrics.consistency_score,
                'completeness_score': metrics.completeness_score
            },
            'distribution_analysis': {
                'difficulty_distribution': metrics.difficulty_distribution,
                'category_distribution': metrics.category_distribution
            },
            'inter_annotator_agreement': {
                'cohens_kappa': metrics.inter_annotator_agreement.cohens_kappa,
                'fleiss_kappa': metrics.inter_annotator_agreement.fleiss_kappa,
                'agreement_percentage': metrics.inter_annotator_agreement.agreement_percentage,
                'correlation_coefficient': metrics.inter_annotator_agreement.correlation_coefficient,
                'sample_size': metrics.inter_annotator_agreement.sample_size,
                'annotator_count': metrics.inter_annotator_agreement.annotator_count,
                'meets_threshold': metrics.inter_annotator_agreement.meets_threshold,
                'confidence_interval': metrics.inter_annotator_agreement.confidence_interval
            },
            'quality_assessment': {
                'issues': metrics.quality_issues,
                'recommendations': metrics.recommendations
            },
            'standards_compliance': {
                'kappa_threshold_met': metrics.inter_annotator_agreement.meets_threshold,
                'minimum_questions_met': metrics.total_questions >= 300,
                'validation_rate_acceptable': metrics.validation_pass_rate >= 0.85,
                'confidence_acceptable': metrics.mean_confidence_score >= 0.7
            }
        }
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Quality report exported to {output_file}")


def create_sample_ratings(questions: List[QuestionItem], num_annotators: int = 3) -> List[AnnotatorRating]:
    """Create sample annotator ratings for testing."""
    ratings = []
    
    annotator_biases = np.random.normal(0, 0.1, num_annotators)  # Small random biases
    
    for question in questions:
        # Base score influenced by difficulty
        if question.difficulty == DifficultyLevel.EASY:
            base_score = 0.8
            noise_std = 0.1
        elif question.difficulty == DifficultyLevel.MEDIUM:
            base_score = 0.7
            noise_std = 0.15
        else:  # HARD
            base_score = 0.6
            noise_std = 0.2
        
        for i in range(num_annotators):
            # Add annotator bias and noise
            score = base_score + annotator_biases[i] + np.random.normal(0, noise_std)
            score = max(0.0, min(1.0, score))  # Clamp to [0,1]
            
            rating = AnnotatorRating(
                annotator_id=f"annotator_{i+1}",
                question_id=question.qid,
                score=score,
                confidence=0.7 + np.random.random() * 0.3,  # 0.7-1.0 confidence
                notes=f"Sample rating for {question.qid}"
            )
            ratings.append(rating)
    
    return ratings
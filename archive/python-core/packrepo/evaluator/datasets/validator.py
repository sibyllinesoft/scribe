"""
Dataset Validator - Advanced Validation and Consistency Checking

Implements comprehensive validation beyond basic schema checking, including:
- Semantic consistency validation 
- Cross-repository question analysis
- Difficulty level verification
- Gold answer quality assessment
- Reference validation and citation checking

Ensures datasets meet the high quality standards required for rigorous
evaluation of PackRepo's token efficiency claims.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
import logging

from .schema import QuestionItem, RepositoryMetadata, DifficultyLevel, QuestionCategory, EvaluationType

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Individual validation issue with severity and context."""
    severity: str  # 'error', 'warning', 'info'
    category: str  # 'schema', 'content', 'consistency', 'reference'
    message: str
    question_id: Optional[str] = None
    context: Dict[str, Any] = None


@dataclass
class ValidationResult:
    """Comprehensive validation results."""
    is_valid: bool
    total_questions: int
    issues: List[ValidationIssue]
    statistics: Dict[str, Any]
    recommendations: List[str]


class AdvancedValidator:
    """Advanced validation beyond basic schema checking."""
    
    def __init__(self):
        """Initialize validator with quality criteria."""
        self.quality_criteria = self._define_quality_criteria()
        self.consistency_checks = self._define_consistency_checks()
    
    def _define_quality_criteria(self) -> Dict[str, Any]:
        """Define advanced quality criteria for validation."""
        return {
            'question_quality': {
                'min_length_words': 5,
                'max_length_words': 50,
                'forbidden_patterns': [
                    r'(?i)\b(todo|fixme|xxx|hack)\b',  # Development artifacts
                    r'(?i)\btest\s*123\b',  # Test data
                    r'(?i)\bplaceholder\b',  # Incomplete content
                ],
                'required_punctuation': ['?'],  # Questions should end with ?
                'clarity_patterns': [
                    r'(?i)\b(what|how|why|when|where|which|who)\b',  # Question words
                    r'(?i)\b(describe|explain|identify|analyze)\b'   # Action verbs
                ]
            },
            'gold_answer_quality': {
                'min_concepts': 2,
                'max_concepts': 10,
                'min_confidence': 0.5,
                'concept_quality_patterns': [
                    r'^[a-zA-Z][a-zA-Z0-9_\-\s]{1,30}$',  # Valid concept format
                ],
                'answer_text_min_length': 10,  # For exact match answers
                'rubric_min_length': 100      # For rubric-based evaluation
            },
            'difficulty_validation': {
                'easy': {
                    'max_question_complexity': 20,  # Word count + concept count
                    'allowed_evaluation_types': ['exact_match', 'regex_match'],
                    'expected_confidence': 0.8
                },
                'medium': {
                    'max_question_complexity': 40,
                    'allowed_evaluation_types': ['semantic', 'rubric_based'],
                    'expected_confidence': 0.7
                },
                'hard': {
                    'max_question_complexity': 100,
                    'allowed_evaluation_types': ['rubric_based', 'semantic'],
                    'expected_confidence': 0.6
                }
            }
        }
    
    def _define_consistency_checks(self) -> Dict[str, Any]:
        """Define consistency checks across questions."""
        return {
            'cross_question_consistency': {
                'max_duplicate_similarity': 0.8,  # Jaccard similarity threshold
                'concept_overlap_threshold': 0.6   # Key concept overlap threshold
            },
            'repository_consistency': {
                'min_questions_per_repo': 3,
                'max_questions_per_repo': 100,
                'concept_diversity_threshold': 0.3  # Min unique concepts per repo
            },
            'category_consistency': {
                'function_behavior_must_have_function': True,
                'api_usage_must_have_class_or_api': True,
                'architecture_questions_require_system_concepts': True
            }
        }
    
    def validate_dataset_comprehensive(
        self,
        questions: List[QuestionItem],
        repository_metadata: List[RepositoryMetadata] = None
    ) -> ValidationResult:
        """
        Perform comprehensive validation of dataset.
        
        Args:
            questions: List of questions to validate
            repository_metadata: Optional repository metadata for cross-validation
            
        Returns:
            Comprehensive validation results
        """
        issues = []
        
        if not questions:
            issues.append(ValidationIssue(
                severity='error',
                category='schema',
                message='Dataset contains no questions'
            ))
            return ValidationResult(
                is_valid=False,
                total_questions=0,
                issues=issues,
                statistics={},
                recommendations=['Add questions to dataset']
            )
        
        logger.info(f"Validating {len(questions)} questions...")
        
        # Individual question validation
        for question in questions:
            question_issues = self._validate_individual_question(question)
            issues.extend(question_issues)
        
        # Cross-question consistency validation
        consistency_issues = self._validate_cross_question_consistency(questions)
        issues.extend(consistency_issues)
        
        # Repository-level validation
        if repository_metadata:
            repo_issues = self._validate_repository_alignment(questions, repository_metadata)
            issues.extend(repo_issues)
        
        # Distribution validation
        distribution_issues = self._validate_distributions(questions)
        issues.extend(distribution_issues)
        
        # Generate statistics
        statistics = self._generate_validation_statistics(questions, issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, statistics)
        
        # Determine overall validity
        error_count = len([i for i in issues if i.severity == 'error'])
        is_valid = error_count == 0
        
        logger.info(f"Validation complete: {error_count} errors, {len(issues)} total issues")
        
        return ValidationResult(
            is_valid=is_valid,
            total_questions=len(questions),
            issues=issues,
            statistics=statistics,
            recommendations=recommendations
        )
    
    def _validate_individual_question(self, question: QuestionItem) -> List[ValidationIssue]:
        """Validate individual question for quality and consistency."""
        issues = []
        
        # Question text validation
        issues.extend(self._validate_question_text(question))
        
        # Gold answer validation
        issues.extend(self._validate_gold_answer(question))
        
        # Difficulty consistency validation
        issues.extend(self._validate_difficulty_consistency(question))
        
        # Category consistency validation
        issues.extend(self._validate_category_consistency(question))
        
        # Evaluation type validation
        issues.extend(self._validate_evaluation_type(question))
        
        return issues
    
    def _validate_question_text(self, question: QuestionItem) -> List[ValidationIssue]:
        """Validate question text quality."""
        issues = []
        criteria = self.quality_criteria['question_quality']
        
        text = question.question.strip()
        words = text.split()
        
        # Length validation
        if len(words) < criteria['min_length_words']:
            issues.append(ValidationIssue(
                severity='warning',
                category='content',
                message=f'Question too short: {len(words)} words < {criteria["min_length_words"]}',
                question_id=question.qid
            ))
        
        if len(words) > criteria['max_length_words']:
            issues.append(ValidationIssue(
                severity='warning',
                category='content',
                message=f'Question too long: {len(words)} words > {criteria["max_length_words"]}',
                question_id=question.qid
            ))
        
        # Forbidden pattern validation
        for pattern in criteria['forbidden_patterns']:
            if re.search(pattern, text):
                issues.append(ValidationIssue(
                    severity='error',
                    category='content',
                    message=f'Question contains forbidden pattern: {pattern}',
                    question_id=question.qid
                ))
        
        # Question format validation
        if not any(punct in text for punct in criteria['required_punctuation']):
            issues.append(ValidationIssue(
                severity='warning',
                category='content',
                message='Question should end with question mark',
                question_id=question.qid
            ))
        
        # Clarity validation
        has_question_word = any(
            re.search(pattern, text) 
            for pattern in criteria['clarity_patterns']
        )
        if not has_question_word:
            issues.append(ValidationIssue(
                severity='info',
                category='content',
                message='Question may lack clarity - consider adding question words',
                question_id=question.qid
            ))
        
        return issues
    
    def _validate_gold_answer(self, question: QuestionItem) -> List[ValidationIssue]:
        """Validate gold answer quality."""
        issues = []
        
        if not question.gold:
            issues.append(ValidationIssue(
                severity='error',
                category='content',
                message='Question missing gold answer',
                question_id=question.qid
            ))
            return issues
        
        criteria = self.quality_criteria['gold_answer_quality']
        gold = question.gold
        
        # Key concepts validation
        if not gold.key_concepts or len(gold.key_concepts) < criteria['min_concepts']:
            issues.append(ValidationIssue(
                severity='error',
                category='content',
                message=f'Insufficient key concepts: {len(gold.key_concepts or [])} < {criteria["min_concepts"]}',
                question_id=question.qid
            ))
        elif len(gold.key_concepts) > criteria['max_concepts']:
            issues.append(ValidationIssue(
                severity='warning',
                category='content',
                message=f'Too many key concepts: {len(gold.key_concepts)} > {criteria["max_concepts"]}',
                question_id=question.qid
            ))
        
        # Concept quality validation
        if gold.key_concepts:
            for concept in gold.key_concepts:
                valid_concept = any(
                    re.match(pattern, concept)
                    for pattern in criteria['concept_quality_patterns']
                )
                if not valid_concept:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='content',
                        message=f'Invalid concept format: "{concept}"',
                        question_id=question.qid
                    ))
        
        # Confidence validation
        if gold.confidence_score < criteria['min_confidence']:
            issues.append(ValidationIssue(
                severity='warning',
                category='content',
                message=f'Low confidence score: {gold.confidence_score}',
                question_id=question.qid
            ))
        
        # Answer text validation (for exact match)
        if question.evaluation_type == EvaluationType.EXACT_MATCH:
            if not gold.answer_text or len(gold.answer_text) < criteria['answer_text_min_length']:
                issues.append(ValidationIssue(
                    severity='error',
                    category='content',
                    message='Exact match question missing adequate answer text',
                    question_id=question.qid
                ))
        
        # Rubric validation (for rubric-based)
        if question.evaluation_type == EvaluationType.RUBRIC_BASED:
            if not question.rubric or len(question.rubric) < criteria['rubric_min_length']:
                issues.append(ValidationIssue(
                    severity='error',
                    category='content',
                    message='Rubric-based question missing adequate rubric',
                    question_id=question.qid
                ))
        
        return issues
    
    def _validate_difficulty_consistency(self, question: QuestionItem) -> List[ValidationIssue]:
        """Validate difficulty level consistency with question characteristics."""
        issues = []
        
        difficulty_criteria = self.quality_criteria['difficulty_validation']
        criteria = difficulty_criteria.get(question.difficulty.value, {})
        
        if not criteria:
            return issues
        
        # Calculate question complexity (simple heuristic)
        question_words = len(question.question.split())
        concept_count = len(question.gold.key_concepts) if question.gold and question.gold.key_concepts else 0
        complexity = question_words + concept_count * 2
        
        if complexity > criteria.get('max_question_complexity', 100):
            issues.append(ValidationIssue(
                severity='warning',
                category='consistency',
                message=f'Question complexity ({complexity}) high for {question.difficulty.value} difficulty',
                question_id=question.qid
            ))
        
        # Evaluation type consistency
        allowed_eval_types = criteria.get('allowed_evaluation_types', [])
        if allowed_eval_types and question.evaluation_type.value not in allowed_eval_types:
            issues.append(ValidationIssue(
                severity='warning',
                category='consistency',
                message=f'Evaluation type {question.evaluation_type.value} unusual for {question.difficulty.value} questions',
                question_id=question.qid
            ))
        
        # Confidence consistency
        expected_confidence = criteria.get('expected_confidence', 0.5)
        if question.gold and question.gold.confidence_score:
            if question.gold.confidence_score < expected_confidence - 0.2:
                issues.append(ValidationIssue(
                    severity='info',
                    category='consistency',
                    message=f'Confidence ({question.gold.confidence_score:.2f}) low for {question.difficulty.value} question',
                    question_id=question.qid
                ))
        
        return issues
    
    def _validate_category_consistency(self, question: QuestionItem) -> List[ValidationIssue]:
        """Validate category consistency with question content."""
        issues = []
        consistency_rules = self.consistency_checks['category_consistency']
        
        question_text = question.question.lower()
        concepts = [c.lower() for c in (question.gold.key_concepts or [])] if question.gold else []
        
        # Function behavior questions should reference functions
        if question.category == QuestionCategory.FUNCTION_BEHAVIOR:
            if consistency_rules['function_behavior_must_have_function']:
                has_function_ref = any(
                    word in question_text or any(word in concept for concept in concepts)
                    for word in ['function', 'method', 'def ', '()', 'returns']
                )
                if not has_function_ref:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='consistency',
                        message='Function behavior question lacks function references',
                        question_id=question.qid
                    ))
        
        # API usage questions should reference classes or APIs
        elif question.category == QuestionCategory.API_USAGE:
            if consistency_rules['api_usage_must_have_class_or_api']:
                has_api_ref = any(
                    word in question_text or any(word in concept for concept in concepts)
                    for word in ['class', 'api', 'interface', 'library', 'module', 'import']
                )
                if not has_api_ref:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='consistency',
                        message='API usage question lacks API/class references',
                        question_id=question.qid
                    ))
        
        # Architecture questions should reference system concepts
        elif question.category == QuestionCategory.ARCHITECTURE_PATTERNS:
            if consistency_rules['architecture_questions_require_system_concepts']:
                has_arch_ref = any(
                    word in question_text or any(word in concept for concept in concepts)
                    for word in ['architecture', 'design', 'pattern', 'system', 'component', 'structure']
                )
                if not has_arch_ref:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='consistency',
                        message='Architecture question lacks architectural concepts',
                        question_id=question.qid
                    ))
        
        return issues
    
    def _validate_evaluation_type(self, question: QuestionItem) -> List[ValidationIssue]:
        """Validate evaluation type consistency with question and answer."""
        issues = []
        
        eval_type = question.evaluation_type
        gold = question.gold
        
        if not gold:
            return issues
        
        # Exact match validation
        if eval_type == EvaluationType.EXACT_MATCH:
            if not gold.answer_text:
                issues.append(ValidationIssue(
                    severity='error',
                    category='consistency',
                    message='Exact match evaluation requires answer_text',
                    question_id=question.qid
                ))
        
        # Regex match validation
        elif eval_type == EvaluationType.REGEX_MATCH:
            if not gold.regex_pattern:
                issues.append(ValidationIssue(
                    severity='error',
                    category='consistency',
                    message='Regex match evaluation requires regex_pattern',
                    question_id=question.qid
                ))
            elif gold.regex_pattern:
                try:
                    re.compile(gold.regex_pattern)
                except re.error:
                    issues.append(ValidationIssue(
                        severity='error',
                        category='consistency',
                        message='Invalid regex pattern in gold answer',
                        question_id=question.qid
                    ))
        
        # Rubric-based validation
        elif eval_type == EvaluationType.RUBRIC_BASED:
            if not question.rubric:
                issues.append(ValidationIssue(
                    severity='error',
                    category='consistency',
                    message='Rubric-based evaluation requires rubric',
                    question_id=question.qid
                ))
        
        return issues
    
    def _validate_cross_question_consistency(self, questions: List[QuestionItem]) -> List[ValidationIssue]:
        """Validate consistency across multiple questions."""
        issues = []
        
        if len(questions) < 2:
            return issues
        
        # Check for duplicate or very similar questions
        for i, q1 in enumerate(questions):
            for j, q2 in enumerate(questions[i+1:], i+1):
                similarity = self._calculate_question_similarity(q1, q2)
                if similarity > self.consistency_checks['cross_question_consistency']['max_duplicate_similarity']:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='consistency',
                        message=f'Questions very similar (similarity: {similarity:.2f})',
                        context={
                            'question_1': q1.qid,
                            'question_2': q2.qid,
                            'similarity': similarity
                        }
                    ))
        
        return issues
    
    def _validate_repository_alignment(
        self, 
        questions: List[QuestionItem], 
        repositories: List[RepositoryMetadata]
    ) -> List[ValidationIssue]:
        """Validate alignment between questions and repository metadata."""
        issues = []
        
        repo_dict = {repo.repo_id: repo for repo in repositories}
        repo_question_counts = {}
        
        # Count questions per repository
        for question in questions:
            repo_id = question.repo_id
            repo_question_counts[repo_id] = repo_question_counts.get(repo_id, 0) + 1
            
            # Validate repository exists
            if repo_id not in repo_dict:
                issues.append(ValidationIssue(
                    severity='error',
                    category='reference',
                    message=f'Question references unknown repository: {repo_id}',
                    question_id=question.qid
                ))
        
        # Validate question distribution per repository
        consistency = self.consistency_checks['repository_consistency']
        for repo_id, question_count in repo_question_counts.items():
            if question_count < consistency['min_questions_per_repo']:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='consistency',
                    message=f'Too few questions for repository {repo_id}: {question_count} < {consistency["min_questions_per_repo"]}',
                    context={'repo_id': repo_id, 'question_count': question_count}
                ))
            
            if question_count > consistency['max_questions_per_repo']:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='consistency',
                    message=f'Too many questions for repository {repo_id}: {question_count} > {consistency["max_questions_per_repo"]}',
                    context={'repo_id': repo_id, 'question_count': question_count}
                ))
        
        return issues
    
    def _validate_distributions(self, questions: List[QuestionItem]) -> List[ValidationIssue]:
        """Validate question distributions meet requirements."""
        issues = []
        
        # Difficulty distribution
        difficulty_counts = {}
        for question in questions:
            diff = question.difficulty.value
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        
        total = len(questions)
        if total > 0:
            easy_pct = difficulty_counts.get('easy', 0) / total
            medium_pct = difficulty_counts.get('medium', 0) / total
            hard_pct = difficulty_counts.get('hard', 0) / total
            
            # Check target distributions (TODO.md requirements)
            if easy_pct < 0.15 or easy_pct > 0.25:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='consistency',
                    message=f'Easy question percentage ({easy_pct:.1%}) outside target range (15-25%)'
                ))
            
            if medium_pct < 0.40 or medium_pct > 0.60:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='consistency',
                    message=f'Medium question percentage ({medium_pct:.1%}) outside target range (40-60%)'
                ))
            
            if hard_pct < 0.25 or hard_pct > 0.35:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='consistency',
                    message=f'Hard question percentage ({hard_pct:.1%}) outside target range (25-35%)'
                ))
        
        return issues
    
    def _calculate_question_similarity(self, q1: QuestionItem, q2: QuestionItem) -> float:
        """Calculate Jaccard similarity between two questions."""
        # Tokenize questions
        tokens1 = set(q1.question.lower().split())
        tokens2 = set(q2.question.lower().split())
        
        # Calculate Jaccard similarity
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _generate_validation_statistics(
        self, 
        questions: List[QuestionItem], 
        issues: List[ValidationIssue]
    ) -> Dict[str, Any]:
        """Generate comprehensive validation statistics."""
        stats = {
            'total_questions': len(questions),
            'issue_summary': {
                'total_issues': len(issues),
                'errors': len([i for i in issues if i.severity == 'error']),
                'warnings': len([i for i in issues if i.severity == 'warning']),
                'info': len([i for i in issues if i.severity == 'info'])
            },
            'issue_categories': {},
            'question_distributions': {},
            'quality_metrics': {}
        }
        
        # Issue category breakdown
        for issue in issues:
            category = issue.category
            stats['issue_categories'][category] = stats['issue_categories'].get(category, 0) + 1
        
        # Question distributions
        if questions:
            difficulty_dist = {}
            category_dist = {}
            eval_type_dist = {}
            
            for q in questions:
                # Difficulty distribution
                diff = q.difficulty.value
                difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1
                
                # Category distribution
                cat = q.category.value
                category_dist[cat] = category_dist.get(cat, 0) + 1
                
                # Evaluation type distribution
                eval_type = q.evaluation_type.value
                eval_type_dist[eval_type] = eval_type_dist.get(eval_type, 0) + 1
            
            stats['question_distributions'] = {
                'difficulty': difficulty_dist,
                'category': category_dist,
                'evaluation_type': eval_type_dist
            }
            
            # Quality metrics
            total = len(questions)
            stats['quality_metrics'] = {
                'questions_with_errors': len(set(
                    i.question_id for i in issues 
                    if i.severity == 'error' and i.question_id
                )),
                'error_rate': len([i for i in issues if i.severity == 'error']) / total,
                'average_confidence': sum(
                    q.gold.confidence_score for q in questions 
                    if q.gold and q.gold.confidence_score
                ) / total if questions else 0.0
            }
        
        return stats
    
    def _generate_recommendations(
        self, 
        issues: List[ValidationIssue], 
        statistics: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        error_count = statistics['issue_summary']['errors']
        warning_count = statistics['issue_summary']['warnings']
        
        if error_count > 0:
            recommendations.append(f"Fix {error_count} critical errors before using dataset")
        
        if warning_count > 10:
            recommendations.append(f"Review {warning_count} warnings to improve dataset quality")
        
        # Category-specific recommendations
        issue_categories = statistics['issue_categories']
        
        if issue_categories.get('content', 0) > 5:
            recommendations.append("Improve question and answer content quality")
        
        if issue_categories.get('consistency', 0) > 5:
            recommendations.append("Review consistency issues across questions")
        
        if issue_categories.get('reference', 0) > 0:
            recommendations.append("Fix reference errors to repositories")
        
        # Quality-specific recommendations
        quality_metrics = statistics.get('quality_metrics', {})
        error_rate = quality_metrics.get('error_rate', 0)
        
        if error_rate > 0.1:  # >10% error rate
            recommendations.append("High error rate indicates systematic quality issues")
        
        avg_confidence = quality_metrics.get('average_confidence', 0)
        if avg_confidence < 0.7:
            recommendations.append("Low average confidence suggests uncertain gold answers")
        
        # Distribution recommendations
        if not recommendations:
            recommendations.append("Dataset passed validation - ready for use")
        
        return recommendations


class DatasetValidator:
    """Main dataset validator interface."""
    
    def __init__(self):
        """Initialize validator."""
        self.advanced_validator = AdvancedValidator()
    
    def validate_dataset_file(self, dataset_path: Path) -> ValidationResult:
        """Validate dataset file comprehensively."""
        questions = []
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Convert to QuestionItem for validation
                        question = self._dict_to_question_item(data)
                        questions.append(question)
                        
                    except (json.JSONDecodeError, ValueError) as e:
                        # Return early with parsing error
                        return ValidationResult(
                            is_valid=False,
                            total_questions=0,
                            issues=[ValidationIssue(
                                severity='error',
                                category='schema',
                                message=f"Line {line_num}: Parse error - {str(e)}"
                            )],
                            statistics={},
                            recommendations=[f"Fix JSON parsing error on line {line_num}"]
                        )
        
        except FileNotFoundError:
            return ValidationResult(
                is_valid=False,
                total_questions=0,
                issues=[ValidationIssue(
                    severity='error',
                    category='schema',
                    message=f"Dataset file not found: {dataset_path}"
                )],
                statistics={},
                recommendations=[f"Create dataset file at {dataset_path}"]
            )
        
        # Perform comprehensive validation
        return self.advanced_validator.validate_dataset_comprehensive(questions)
    
    def _dict_to_question_item(self, data: Dict[str, Any]) -> QuestionItem:
        """Convert dictionary to QuestionItem for validation."""
        from .schema import GoldAnswer, QuestionCategory, DifficultyLevel, EvaluationType
        
        # Extract gold answer
        gold_data = data.get('gold', {})
        gold = GoldAnswer(
            answer_text=gold_data.get('answer_text'),
            regex_pattern=gold_data.get('regex_pattern'),
            key_concepts=gold_data.get('key_concepts', []),
            evaluation_rubric=gold_data.get('evaluation_rubric'),
            confidence_score=gold_data.get('confidence_score', 1.0),
            annotator_id=gold_data.get('annotator_id', 'unknown'),
            validation_notes=gold_data.get('validation_notes', '')
        )
        
        return QuestionItem(
            repo_id=data.get('repo_id', 'unknown'),
            qid=data.get('qid', 'unknown'),
            question=data.get('question', ''),
            category=QuestionCategory(data.get('category', 'function_behavior')),
            difficulty=DifficultyLevel(data.get('difficulty', 'medium')),
            evaluation_type=EvaluationType(data.get('evaluation_type', 'semantic')),
            pack_budget=data.get('pack_budget', 10000),
            gold=gold,
            rubric=data.get('rubric'),
            metadata=data.get('metadata', {})
        )
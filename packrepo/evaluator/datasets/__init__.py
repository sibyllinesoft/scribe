"""
PackRepo Dataset Framework

This module provides comprehensive dataset creation, curation, and validation
for the PackRepo evaluation system, meeting the rigorous requirements outlined
in TODO.md for ground truth QA evaluation.

Key Features:
- Repository selection and curation pipeline
- Question generation with difficulty distribution
- Gold answer creation and validation  
- Inter-annotator agreement measurement (κ≥0.6 required)
- QA dataset structure in JSON Lines format
- Automated quality assurance and consistency checking

Components:
- DatasetCurator: Repository selection and metadata extraction
- QuestionGenerator: Question creation with proper categorization
- GoldAnswerValidator: Ground truth validation and rubric creation
- QualityAssurance: Inter-annotator agreement and consistency checking
- DatasetBuilder: Main orchestrator for dataset creation pipeline
"""

from .curator import DatasetCurator, RepositoryMetadata
from .generator import QuestionGenerator, QuestionItem, DifficultyLevel, QuestionCategory
from .validator import GoldAnswerValidator, GoldAnswer, EvaluationRubric
from .quality import QualityAssurance, InterAnnotatorAgreement
from .builder import DatasetBuilder
from .schema import QADatasetSchema, validate_dataset

__all__ = [
    'DatasetCurator',
    'RepositoryMetadata', 
    'QuestionGenerator',
    'QuestionItem',
    'DifficultyLevel',
    'QuestionCategory',
    'GoldAnswerValidator',
    'GoldAnswer', 
    'EvaluationRubric',
    'QualityAssurance',
    'InterAnnotatorAgreement',
    'DatasetBuilder',
    'QADatasetSchema',
    'validate_dataset'
]
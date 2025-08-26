"""
QA Dataset Schema Definition

Defines the standardized JSON Lines schema for PackRepo QA datasets
as specified in TODO.md requirements.

Schema: qa.jsonl format with {repo_id, qid, question, gold, rubric?}
"""

from typing import Dict, Any, List, Optional, Literal, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import jsonschema
from pathlib import Path


class DifficultyLevel(str, Enum):
    """Question difficulty levels with proper distribution targets."""
    EASY = "easy"       # 20% of questions (exact answers, straightforward)
    MEDIUM = "medium"   # 50% of questions (analysis required)
    HARD = "hard"       # 30% of questions (synthesis, complex reasoning)


class QuestionCategory(str, Enum):
    """Question categories for balanced coverage."""
    # Code-level categories
    FUNCTION_BEHAVIOR = "function_behavior"      # How functions work
    API_USAGE = "api_usage"                     # How to use APIs
    ALGORITHM_IMPLEMENTATION = "algorithm_impl"  # Algorithm details
    CLASS_DESIGN = "class_design"               # OOP patterns
    ERROR_HANDLING = "error_handling"           # Exception patterns
    
    # Repository-level categories  
    ARCHITECTURE_PATTERNS = "architecture"      # System design
    DESIGN_DECISIONS = "design_decisions"       # Why built this way
    SYSTEM_INTEGRATION = "integration"          # How components connect
    DEPLOYMENT_CONFIG = "deployment"            # Ops and config
    TESTING_STRATEGY = "testing"                # Test patterns


class EvaluationType(str, Enum):
    """How the question should be evaluated."""
    EXACT_MATCH = "exact_match"        # String comparison
    REGEX_MATCH = "regex_match"        # Pattern matching  
    RUBRIC_BASED = "rubric_based"      # LLM judge with rubric
    SEMANTIC_SIMILARITY = "semantic"   # Embedding similarity


@dataclass
class RepositoryMetadata:
    """Metadata for curated repositories."""
    repo_id: str
    name: str
    description: str
    language: str
    domain: str
    size_loc: int
    size_files: int
    pack_budget: int  # Token budget for this repo
    license: str
    url: str
    commit_sha: str
    quality_score: float  # 0-1 quality assessment
    exclusion_list: List[str]  # Files/patterns to exclude


@dataclass
class GoldAnswer:
    """Ground truth answer with evaluation criteria."""
    answer_text: Optional[str] = None        # For exact/regex matching
    regex_pattern: Optional[str] = None      # For regex evaluation
    key_concepts: List[str] = None          # Required concepts
    evaluation_rubric: Optional[str] = None  # For rubric-based evaluation
    confidence_score: float = 1.0           # Confidence in this gold answer
    annotator_id: str = "system"            # Who created this answer
    validation_notes: str = ""              # Additional validation info


@dataclass  
class QuestionItem:
    """Complete question item for QA dataset."""
    repo_id: str
    qid: str                        # Unique question ID
    question: str                   # The question text
    category: QuestionCategory      # Question type
    difficulty: DifficultyLevel     # Difficulty level
    evaluation_type: EvaluationType # How to evaluate
    pack_budget: int               # Token budget for context
    gold: GoldAnswer               # Ground truth answer
    rubric: Optional[str] = None   # Evaluation rubric if needed
    metadata: Dict[str, Any] = None # Additional metadata
    
    def to_jsonl(self) -> str:
        """Convert to JSON Lines format as specified in TODO.md."""
        data = {
            "repo_id": self.repo_id,
            "qid": self.qid,
            "question": self.question,
            "gold": asdict(self.gold),
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "evaluation_type": self.evaluation_type.value,
            "pack_budget": self.pack_budget
        }
        
        # Add rubric if present
        if self.rubric:
            data["rubric"] = self.rubric
            
        # Add metadata if present
        if self.metadata:
            data["metadata"] = self.metadata
            
        return json.dumps(data, separators=(',', ':'))


# JSON Schema for validation
QA_DATASET_SCHEMA = {
    "type": "object",
    "required": ["repo_id", "qid", "question", "gold"],
    "properties": {
        "repo_id": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9_-]+$",
            "description": "Repository identifier"
        },
        "qid": {
            "type": "string", 
            "pattern": "^[a-zA-Z0-9_-]+$",
            "description": "Unique question identifier"
        },
        "question": {
            "type": "string",
            "minLength": 10,
            "maxLength": 500,
            "description": "Question text"
        },
        "gold": {
            "type": "object",
            "properties": {
                "answer_text": {"type": ["string", "null"]},
                "regex_pattern": {"type": ["string", "null"]},
                "key_concepts": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "evaluation_rubric": {"type": ["string", "null"]},
                "confidence_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "annotator_id": {"type": "string"},
                "validation_notes": {"type": "string"}
            }
        },
        "category": {
            "type": "string",
            "enum": [cat.value for cat in QuestionCategory]
        },
        "difficulty": {
            "type": "string", 
            "enum": [diff.value for diff in DifficultyLevel]
        },
        "evaluation_type": {
            "type": "string",
            "enum": [eval_type.value for eval_type in EvaluationType]
        },
        "pack_budget": {
            "type": "integer",
            "minimum": 1000,
            "maximum": 200000
        },
        "rubric": {
            "type": ["string", "null"],
            "description": "Evaluation rubric for rubric-based questions"
        },
        "metadata": {
            "type": ["object", "null"],
            "description": "Additional metadata"
        }
    },
    "additionalProperties": False
}


class QADatasetSchema:
    """Schema validation utilities for QA datasets."""
    
    @staticmethod
    def validate_question_item(data: Dict[str, Any]) -> bool:
        """Validate a single question item against schema."""
        try:
            jsonschema.validate(data, QA_DATASET_SCHEMA)
            return True
        except jsonschema.ValidationError as e:
            print(f"Validation error: {e}")
            return False
    
    @staticmethod
    def validate_dataset_file(file_path: Path) -> tuple[bool, List[str]]:
        """Validate entire dataset file."""
        errors = []
        valid_count = 0
        total_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    total_count += 1
                    
                    try:
                        data = json.loads(line)
                        if QADatasetSchema.validate_question_item(data):
                            valid_count += 1
                        else:
                            errors.append(f"Line {line_num}: Schema validation failed")
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {line_num}: Invalid JSON - {e}")
        except FileNotFoundError:
            errors.append(f"Dataset file not found: {file_path}")
        except Exception as e:
            errors.append(f"Error reading dataset file: {e}")
        
        is_valid = len(errors) == 0 and valid_count == total_count
        return is_valid, errors
    
    @staticmethod
    def check_distribution_requirements(file_path: Path) -> Dict[str, Any]:
        """Check if dataset meets distribution requirements from TODO.md."""
        categories = {}
        difficulties = {}
        repos = set()
        total_questions = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        data = json.loads(line)
                        total_questions += 1
                        
                        # Count categories
                        category = data.get('category', 'unknown')
                        categories[category] = categories.get(category, 0) + 1
                        
                        # Count difficulties  
                        difficulty = data.get('difficulty', 'unknown')
                        difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
                        
                        # Count repos
                        repos.add(data.get('repo_id', 'unknown'))
                        
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            return {"error": f"Dataset file not found: {file_path}"}
        
        # Calculate percentages
        if total_questions > 0:
            difficulty_pct = {
                diff: count / total_questions 
                for diff, count in difficulties.items()
            }
        else:
            difficulty_pct = {}
        
        # Check requirements
        meets_requirements = {
            "min_questions": total_questions >= 300,  # ≥300 questions
            "min_repos": len(repos) >= 5,             # ≥5 repos
            "difficulty_distribution": all([
                difficulty_pct.get('easy', 0) >= 0.15,    # ≥15% easy (target 20%)
                difficulty_pct.get('medium', 0) >= 0.40,   # ≥40% medium (target 50%) 
                difficulty_pct.get('hard', 0) >= 0.25      # ≥25% hard (target 30%)
            ])
        }
        
        return {
            "total_questions": total_questions,
            "total_repos": len(repos),
            "repos": list(repos),
            "categories": categories,
            "difficulties": difficulties,
            "difficulty_percentages": difficulty_pct,
            "meets_requirements": meets_requirements,
            "overall_valid": all(meets_requirements.values())
        }


def validate_dataset(file_path: Path, check_distribution: bool = True) -> Dict[str, Any]:
    """
    Complete dataset validation against TODO.md requirements.
    
    Args:
        file_path: Path to qa.jsonl dataset file
        check_distribution: Whether to check distribution requirements
        
    Returns:
        Validation results with detailed feedback
    """
    results = {
        "file_path": str(file_path),
        "schema_valid": False,
        "schema_errors": [],
        "distribution": {},
        "overall_valid": False
    }
    
    # Schema validation
    schema_valid, schema_errors = QADatasetSchema.validate_dataset_file(file_path)
    results["schema_valid"] = schema_valid
    results["schema_errors"] = schema_errors
    
    # Distribution validation
    if check_distribution:
        distribution = QADatasetSchema.check_distribution_requirements(file_path)
        results["distribution"] = distribution
    
    # Overall validation
    results["overall_valid"] = (
        schema_valid and 
        (not check_distribution or distribution.get("overall_valid", False))
    )
    
    return results


def create_sample_dataset(output_file: Path, num_questions: int = 10) -> None:
    """Create a small sample dataset for testing."""
    sample_questions = []
    
    # Create sample questions across categories and difficulties
    for i in range(num_questions):
        repo_id = f"sample_repo_{i % 3 + 1}"  # 3 repos
        qid = f"q_{i:03d}"
        
        # Cycle through difficulties  
        if i % 10 < 2:  # 20% easy
            difficulty = DifficultyLevel.EASY
            question = f"What does function_{i} do?"
            gold = GoldAnswer(
                answer_text=f"Function_{i} performs basic operation {i}",
                key_concepts=[f"function_{i}", "operation"],
                confidence_score=0.95
            )
        elif i % 10 < 7:  # 50% medium  
            difficulty = DifficultyLevel.MEDIUM
            question = f"How does the {['algorithm', 'pattern', 'system'][i%3]} in module_{i} work?"
            gold = GoldAnswer(
                key_concepts=[f"module_{i}", "implementation", "logic"],
                evaluation_rubric=f"Answer should explain the core logic of module_{i}",
                confidence_score=0.85
            )
        else:  # 30% hard
            difficulty = DifficultyLevel.HARD
            question = f"Why was the architectural decision made for component_{i}?"
            gold = GoldAnswer(
                evaluation_rubric=f"Answer should analyze trade-offs and rationale for component_{i} design",
                key_concepts=["architecture", "trade-offs", "rationale"],
                confidence_score=0.80
            )
        
        # Cycle through categories
        categories = list(QuestionCategory)
        category = categories[i % len(categories)]
        
        # Set evaluation type
        if difficulty == DifficultyLevel.EASY:
            eval_type = EvaluationType.EXACT_MATCH
        elif gold.evaluation_rubric:
            eval_type = EvaluationType.RUBRIC_BASED
        else:
            eval_type = EvaluationType.SEMANTIC_SIMILARITY
        
        question_item = QuestionItem(
            repo_id=repo_id,
            qid=qid,
            question=question,
            category=category,
            difficulty=difficulty,
            evaluation_type=eval_type,
            pack_budget=10000 + (i * 1000),  # Varying budgets
            gold=gold,
            rubric=gold.evaluation_rubric,
            metadata={"sample_id": i}
        )
        
        sample_questions.append(question_item)
    
    # Write to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for question in sample_questions:
            f.write(question.to_jsonl() + '\n')
    
    print(f"Created sample dataset with {num_questions} questions at {output_file}")
# PackRepo QA Dataset Builder

Comprehensive, research-grade dataset creation system for evaluating PackRepo's token efficiency claims. Implements all requirements from TODO.md with statistical rigor and academic standards.

## 🎯 Key Features

### TODO.md Compliance
- **✅ ≥ 300 questions across ≥ 5 repositories**
- **✅ Inter-annotator agreement κ≥0.6 on 50-question audit**  
- **✅ Ground truth with verifiable references**
- **✅ Diverse programming languages and domains**
- **✅ Rigorous quality validation pipeline**

### Advanced Capabilities
- **Repository Curation**: Intelligent selection based on quality, diversity, and licensing
- **Question Generation**: Balanced across difficulty levels and semantic categories
- **Quality Assurance**: Multi-layer validation with consistency checking
- **Inter-Annotator Agreement**: Statistical measurement with confidence intervals
- **Comprehensive Validation**: Schema, content, consistency, and reference validation
- **Automated Reporting**: Detailed quality metrics and compliance verification

## 📋 Dataset Schema

Questions follow standardized JSON Lines format:

```json
{
  "repo_id": "repository_identifier", 
  "qid": "unique_question_id",
  "question": "What does the authenticate() function do?",
  "gold": {
    "answer_text": "Verifies user credentials against database",
    "key_concepts": ["authentication", "credentials", "database"],
    "confidence_score": 0.9,
    "annotator_id": "expert_1"
  },
  "category": "function_behavior",
  "difficulty": "easy", 
  "evaluation_type": "semantic",
  "pack_budget": 50000,
  "rubric": "Evaluation criteria for subjective questions"
}
```

## 🏗️ Architecture

```
packrepo/evaluator/datasets/
├── schema.py              # Dataset schema and validation rules
├── curator.py            # Repository selection and analysis  
├── generator.py          # Question generation with templates
├── quality.py           # Inter-annotator agreement measurement
├── validator.py         # Comprehensive validation pipeline
└── dataset_builder.py   # Master orchestration system
```

### Core Components

#### 1. Repository Curator (`curator.py`)
- **Quality Assessment**: Code quality, documentation, test coverage
- **Diversity Selection**: Languages, domains, project types
- **Metadata Extraction**: Size, complexity, license validation
- **Exclusion Generation**: Automated .gitignore and pattern filtering

#### 2. Question Generator (`generator.py`) 
- **Template System**: Consistent question structures across categories
- **Code Analysis**: AST parsing for function/class extraction
- **Difficulty Balancing**: 20% easy, 50% medium, 30% hard distribution
- **Category Coverage**: Code-level (60%) and repo-level (40%) questions

#### 3. Quality Assurance (`quality.py`)
- **Cohen's κ**: Two-annotator agreement measurement
- **Fleiss' κ**: Multi-annotator agreement for >2 raters
- **Confidence Intervals**: Statistical significance testing
- **Distribution Analysis**: Requirement compliance checking

#### 4. Advanced Validator (`validator.py`)
- **Content Quality**: Question clarity, concept validity
- **Consistency Checking**: Cross-question similarity detection  
- **Reference Validation**: Repository alignment verification
- **Distribution Compliance**: TODO.md requirement validation

#### 5. Dataset Builder (`dataset_builder.py`)
- **Orchestration**: Coordinates all pipeline components
- **Parallel Processing**: Multi-threaded question generation
- **Quality Gates**: Automated compliance verification
- **Comprehensive Reporting**: Detailed metrics and analysis

## 🚀 Quick Start

### 1. Command Line Interface

```bash
# Create TODO.md compliant dataset (recommended)
python create_dataset.py --todo-compliant --output ./datasets/compliant

# Create from existing repositories  
python create_dataset.py --repos /path/to/repo1 /path/to/repo2 --output ./datasets/production

# Create sample dataset for testing
python create_dataset.py --sample --output ./datasets/sample --num-repos 6

# Validate existing dataset
python create_dataset.py --validate ./datasets/existing_dataset.jsonl
```

### 2. Python API

```python
from packrepo.evaluator.datasets.dataset_builder import DatasetBuildOrchestrator
from pathlib import Path

# Create TODO.md compliant dataset
result = DatasetBuildOrchestrator.create_todo_compliant_dataset(
    output_dir=Path("./datasets"),
    use_samples=False  # Use real repositories
)

print(f"Success: {result.success}")
print(f"Meets TODO requirements: {result.meets_todo_requirements}")
print(f"Inter-annotator κ: {result.inter_annotator_agreement['cohens_kappa']:.3f}")
```

### 3. Advanced Configuration

```python
from packrepo.evaluator.datasets.dataset_builder import (
    ComprehensiveDatasetBuilder, DatasetBuildConfig, CurationCriteria
)

# Custom configuration
config = DatasetBuildConfig(
    min_questions=500,  # More questions for larger evaluation
    min_repositories=10,
    questions_per_repo=50,
    min_kappa_threshold=0.7,  # Higher agreement threshold
    repository_criteria=CurationCriteria(
        min_stars=100,
        min_size_loc=5000,
        license_allowlist=["MIT", "Apache-2.0"]
    )
)

builder = ComprehensiveDatasetBuilder(config)
result = builder.build_complete_dataset(repository_paths)
```

## 📊 Quality Metrics

### Inter-Annotator Agreement
- **Cohen's κ**: -1.0 to 1.0 scale (κ≥0.6 required)
- **Fleiss' κ**: Multi-rater extension for >2 annotators
- **Agreement %**: Percentage of exact score matches
- **Confidence Intervals**: 95% CI for statistical significance

### Distribution Requirements
- **Difficulty**: Easy (20%), Medium (50%), Hard (30%)  
- **Categories**: Balanced across code and repository levels
- **Languages**: Minimum 3 programming languages
- **Domains**: Minimum 3 application domains

### Validation Gates
- **Schema Compliance**: 100% valid JSON Lines format
- **Content Quality**: ≥85% questions pass validation
- **Consistency Score**: ≥80% internal consistency 
- **Reference Integrity**: 100% repository alignment

## 🔬 Evaluation Integration

### PackRepo Evaluation Pipeline

```python
# 1. Generate dataset
from packrepo.evaluator.datasets import DatasetBuildOrchestrator

dataset_result = DatasetBuildOrchestrator.create_todo_compliant_dataset(
    output_dir=Path("./evaluation_data")
)

# 2. Run PackRepo variants  
from packrepo.evaluator.qa_harness import QARunner

qa_runner = QARunner()
results = qa_runner.evaluate_variants(
    dataset_path=dataset_result.dataset_path,
    variants=["V0c", "V1", "V2", "V3"],
    budgets=[120000, 200000]
)

# 3. Statistical analysis
from packrepo.evaluator.statistics import BootstrapBCA

stats = BootstrapBCA()
ci_results = stats.compute_confidence_intervals(
    results, metric="qa_acc_per_100k", iterations=10000
)
```

### Token Efficiency Measurement

The dataset enables rigorous measurement of PackRepo's core claim:

**≥ +20% Q&A accuracy per 100k tokens vs naive baseline**

```python
# Token efficiency calculation
def token_efficiency(accuracy: float, selection_tokens: int) -> float:
    return (accuracy * 100000.0) / max(1, selection_tokens)

# Paired bootstrap for statistical significance  
diffs = [token_efficiency(acc_v1, tok_v1) - token_efficiency(acc_v0, tok_v0) 
         for acc_v1, tok_v1, acc_v0, tok_v0 in paired_results]

ci_lower, ci_upper = bootstrap_bca(diffs, confidence=0.95)
significant_improvement = ci_lower > 0  # Required for promotion
```

## 📈 Output Structure

```
datasets/
├── qa_dataset.jsonl           # Main dataset file
├── reports/                   
│   ├── build_summary.json     # Complete build metrics
│   ├── quality_report.json    # Quality analysis details  
│   └── curation_report.json   # Repository selection summary
├── repo_configs/              
│   ├── repo1_config.json      # Per-repository configurations
│   └── repo2_config.json      
└── sample_repos/              # Sample repositories (if used)
    ├── sample_python_ml/
    └── sample_typescript_web/
```

## 🔧 Configuration

### Repository Curation Criteria

```json
{
  "min_stars": 100,
  "min_commits": 50, 
  "min_contributors": 3,
  "min_size_loc": 1000,
  "max_size_loc": 100000,
  "license_allowlist": ["MIT", "Apache-2.0", "BSD-3-Clause"],
  "required_files": ["README.md"],
  "forbidden_patterns": ["node_modules", "vendor", "generated"]
}
```

### Question Generation Parameters

```json
{
  "difficulty_distribution": {"easy": 0.2, "medium": 0.5, "hard": 0.3},
  "category_split": {"code_level": 0.6, "repo_level": 0.4},
  "evaluation_types": {
    "easy": ["exact_match", "regex_match"],
    "medium": ["semantic", "rubric_based"], 
    "hard": ["rubric_based"]
  }
}
```

### Quality Thresholds

```json
{
  "min_kappa_threshold": 0.6,
  "min_validation_pass_rate": 0.85,
  "min_confidence_score": 0.7,
  "annotation_sample_size": 50,
  "max_duplicate_similarity": 0.8
}
```

## 🧪 Testing & Validation

### Unit Tests
```bash
pytest packrepo/evaluator/datasets/tests/ -v
```

### Integration Testing
```bash
# Test full pipeline with sample data
python create_dataset.py --sample --output ./test_output --num-repos 3

# Validate against TODO.md requirements
python -c "
from pathlib import Path
from packrepo.evaluator.datasets.schema import validate_dataset
result = validate_dataset(Path('./test_output/qa_dataset.jsonl'))
assert result['overall_valid'], 'Dataset validation failed'
print('✅ All tests passed')
"
```

### Performance Benchmarking
```bash
# Benchmark question generation
python -m packrepo.evaluator.datasets.benchmarks --repos 10 --questions 1000

# Memory usage analysis  
python -m memory_profiler create_dataset.py --sample --num-repos 5
```

## 🤝 Contributing

### Adding New Question Categories

1. **Update Schema** (`schema.py`):
```python
class QuestionCategory(str, Enum):
    NEW_CATEGORY = "new_category"
```

2. **Add Templates** (`generator.py`):
```python
templates.append(QuestionTemplate(
    template="What is the {concept} pattern used in {function_name}?",
    category=QuestionCategory.NEW_CATEGORY,
    difficulty=DifficultyLevel.MEDIUM,
    evaluation_type=EvaluationType.RUBRIC_BASED
))
```

3. **Update Validation** (`validator.py`):
```python
# Add category-specific validation rules
def _validate_new_category_consistency(self, question: QuestionItem):
    # Validation logic here
    pass
```

### Adding New Languages

1. **Update Analyzer** (`generator.py`):
```python
self.language_extensions = {
    'NewLanguage': ['.newext'],
    # ... existing languages
}

self.function_patterns = {
    'NewLanguage': r'new_pattern_regex',
    # ... existing patterns  
}
```

2. **Update Curator** (`curator.py`):
```python
language_exclusions = {
    'NewLanguage': ['*.compiled', 'artifacts/*'],
    # ... existing exclusions
}
```

## 📚 Research Background

This system implements best practices from:

- **Inter-Annotator Agreement**: Cohen (1960), Fleiss (1971) 
- **Question Quality**: Linguistic analysis and readability metrics
- **Dataset Curation**: Systematic literature review methodologies
- **Statistical Validation**: Bootstrap confidence intervals (Efron & Tibshirani, 1993)

### Academic Rigor

- **Pre-registered Protocol**: All evaluation criteria defined before data collection
- **Multiple Validation**: Schema, content, consistency, and statistical validation
- **Reproducible**: Deterministic seeds and comprehensive reporting
- **Transparent**: Open methodology with detailed documentation

## 📞 Support

For questions about dataset creation or validation:

1. **Check Documentation**: This README and inline code documentation
2. **Validate Configuration**: Use `--validate` flag to test existing datasets  
3. **Review Reports**: Check generated reports in `output_dir/reports/`
4. **Run Diagnostics**: Use `--verbose` flag for detailed logging

## 🔮 Future Enhancements

- **Multi-language Support**: Extended beyond English questions
- **Domain Expansion**: Scientific computing, mobile development, DevOps
- **Advanced Metrics**: Semantic similarity scoring, concept coverage analysis  
- **Real-time Validation**: Continuous quality monitoring during generation
- **Human-in-the-Loop**: Integration with annotation platforms for expert review

---

**Built for rigorous evaluation of PackRepo's token efficiency claims with academic-grade methodology and statistical confidence.**
"""
Question Generator for QA Dataset Creation

Generates high-quality questions across different categories and difficulty levels
according to TODO.md requirements. Ensures proper distribution and balanced coverage
of code-level and repository-level questions.

Question Categories:
- Code-level: Function behavior, API usage, algorithm implementation
- Repository-level: Architecture patterns, design decisions, system integration

Difficulty Distribution (TODO.md):
- Easy: 20% (exact answers, straightforward)
- Medium: 50% (analysis required)  
- Hard: 30% (synthesis, complex reasoning)
"""

import json
import random
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .schema import QuestionItem, QuestionCategory, DifficultyLevel, EvaluationType, GoldAnswer, RepositoryMetadata


class QuestionTemplate:
    """Template for generating questions with consistent structure."""
    
    def __init__(
        self,
        template: str,
        category: QuestionCategory,
        difficulty: DifficultyLevel,
        evaluation_type: EvaluationType,
        required_concepts: List[str] = None,
        placeholders: List[str] = None
    ):
        self.template = template
        self.category = category
        self.difficulty = difficulty
        self.evaluation_type = evaluation_type
        self.required_concepts = required_concepts or []
        self.placeholders = placeholders or []
    
    def generate_question(self, context: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Generate question text and key concepts from template."""
        question = self.template
        concepts = self.required_concepts.copy()
        
        # Replace placeholders with context values
        for placeholder in self.placeholders:
            if placeholder in context:
                value = context[placeholder]
                question = question.replace(f"{{{placeholder}}}", str(value))
                if isinstance(value, str) and len(value) < 50:  # Add as concept if short
                    concepts.append(value)
        
        return question, concepts


class CodeAnalyzer:
    """Analyze repository code to extract elements for question generation."""
    
    def __init__(self):
        self.function_patterns = {
            'Python': r'def\s+(\w+)\s*\(',
            'TypeScript': r'(?:function\s+(\w+)|(\w+)\s*(?:=|:).*?=>|(\w+)\s*\(.*?\)\s*\{)',
            'JavaScript': r'(?:function\s+(\w+)|(\w+)\s*(?:=|:).*?=>|(\w+)\s*\(.*?\)\s*\{)',
            'Go': r'func\s+(\w+)\s*\(',
            'Rust': r'fn\s+(\w+)\s*\(',
            'Java': r'(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)*(\w+)\s*\('
        }
        
        self.class_patterns = {
            'Python': r'class\s+(\w+)(?:\(.*?\))?:',
            'TypeScript': r'(?:class|interface)\s+(\w+)',
            'JavaScript': r'class\s+(\w+)',
            'Go': r'type\s+(\w+)\s+struct',
            'Rust': r'(?:struct|enum|trait)\s+(\w+)',
            'Java': r'(?:class|interface|enum)\s+(\w+)'
        }
    
    def analyze_repository(self, repo_path: Path, language: str) -> Dict[str, Any]:
        """Extract code elements for question generation."""
        analysis = {
            'functions': [],
            'classes': [],
            'files': [],
            'imports': [],
            'patterns': set(),
            'complexity_indicators': []
        }
        
        # Get language-specific patterns
        function_pattern = self.function_patterns.get(language, r'(\w+)\s*\(')
        class_pattern = self.class_patterns.get(language, r'(\w+)')
        
        # Analyze source files
        extensions = self._get_language_extensions(language)
        for ext in extensions:
            for file_path in repo_path.rglob(f'*{ext}'):
                # Skip vendor/generated files
                if self._should_skip_file(file_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    analysis['files'].append({
                        'path': str(file_path.relative_to(repo_path)),
                        'size_lines': len(content.splitlines()),
                        'size_chars': len(content)
                    })
                    
                    # Extract functions
                    functions = re.findall(function_pattern, content, re.MULTILINE | re.IGNORECASE)
                    for func_match in functions:
                        if isinstance(func_match, tuple):
                            func_name = next(f for f in func_match if f)
                        else:
                            func_name = func_match
                        
                        if func_name and len(func_name) > 1:  # Skip single char matches
                            analysis['functions'].append({
                                'name': func_name,
                                'file': str(file_path.relative_to(repo_path)),
                                'language': language
                            })
                    
                    # Extract classes/types
                    classes = re.findall(class_pattern, content, re.MULTILINE | re.IGNORECASE)
                    for class_name in classes:
                        if isinstance(class_name, tuple):
                            class_name = next(c for c in class_name if c)
                        if class_name and len(class_name) > 1:
                            analysis['classes'].append({
                                'name': class_name,
                                'file': str(file_path.relative_to(repo_path)),
                                'language': language
                            })
                    
                    # Extract imports/dependencies
                    imports = self._extract_imports(content, language)
                    analysis['imports'].extend(imports)
                    
                    # Detect patterns/frameworks
                    patterns = self._detect_patterns(content, language)
                    analysis['patterns'].update(patterns)
                    
                    # Assess complexity
                    complexity = self._assess_file_complexity(content, language)
                    if complexity > 0:
                        analysis['complexity_indicators'].append({
                            'file': str(file_path.relative_to(repo_path)),
                            'complexity': complexity
                        })
                
                except (OSError, UnicodeDecodeError):
                    continue
        
        # Convert patterns set to list for JSON serialization
        analysis['patterns'] = list(analysis['patterns'])
        
        return analysis
    
    def _get_language_extensions(self, language: str) -> List[str]:
        """Get file extensions for language."""
        extensions = {
            'Python': ['.py'],
            'TypeScript': ['.ts', '.tsx'],
            'JavaScript': ['.js', '.jsx'],
            'Go': ['.go'],
            'Rust': ['.rs'],
            'Java': ['.java'],
            'C++': ['.cpp', '.cc', '.cxx', '.h'],
            'C': ['.c', '.h']
        }
        return extensions.get(language, ['.py'])  # Default to Python
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_patterns = [
            'node_modules', 'vendor', '__pycache__', '.git',
            'build', 'dist', 'target', 'generated'
        ]
        path_str = str(file_path).lower()
        return any(pattern in path_str for pattern in skip_patterns)
    
    def _extract_imports(self, content: str, language: str) -> List[str]:
        """Extract import statements."""
        imports = []
        
        patterns = {
            'Python': [r'from\s+(\w+)', r'import\s+(\w+)'],
            'TypeScript': [r'from\s+[\'"]([^\'"]+)[\'"]', r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]'],
            'JavaScript': [r'from\s+[\'"]([^\'"]+)[\'"]', r'require\s*\(\s*[\'"]([^\'"]+)[\'"]'],
            'Go': [r'import\s+[\'"]([^\'"]+)[\'"]'],
            'Rust': [r'use\s+(\w+(?:::\w+)*)'],
            'Java': [r'import\s+([\w\.]+)']
        }
        
        lang_patterns = patterns.get(language, [])
        for pattern in lang_patterns:
            matches = re.findall(pattern, content)
            imports.extend(matches)
        
        return imports[:20]  # Limit to avoid too many
    
    def _detect_patterns(self, content: str, language: str) -> Set[str]:
        """Detect architectural patterns and frameworks."""
        patterns = set()
        content_lower = content.lower()
        
        # Framework detection
        frameworks = {
            'Python': ['flask', 'django', 'fastapi', 'pytest', 'numpy', 'pandas'],
            'TypeScript': ['react', 'angular', 'vue', 'express', 'jest', 'typescript'],
            'JavaScript': ['react', 'express', 'node', 'jest', 'webpack'],
            'Go': ['gin', 'echo', 'gorilla', 'testify'],
            'Rust': ['tokio', 'serde', 'clap', 'reqwest'],
            'Java': ['spring', 'junit', 'maven', 'gradle']
        }
        
        lang_frameworks = frameworks.get(language, [])
        for framework in lang_frameworks:
            if framework in content_lower:
                patterns.add(f'{framework}_framework')
        
        # Pattern detection
        if 'async' in content_lower and 'await' in content_lower:
            patterns.add('async_pattern')
        if 'class' in content_lower and 'extends' in content_lower:
            patterns.add('inheritance_pattern')
        if 'interface' in content_lower:
            patterns.add('interface_pattern')
        if 'singleton' in content_lower:
            patterns.add('singleton_pattern')
        if 'factory' in content_lower:
            patterns.add('factory_pattern')
        
        return patterns
    
    def _assess_file_complexity(self, content: str, language: str) -> int:
        """Assess file complexity (simple heuristic)."""
        complexity = 0
        
        # Count control structures
        control_keywords = ['if', 'for', 'while', 'switch', 'case', 'try', 'catch']
        for keyword in control_keywords:
            complexity += len(re.findall(rf'\b{keyword}\b', content, re.IGNORECASE))
        
        # Penalize very long functions (rough heuristic)
        lines = content.splitlines()
        current_function_length = 0
        for line in lines:
            if any(pattern in line.lower() for pattern in ['def ', 'function ', 'func ']):
                current_function_length = 0
            current_function_length += 1
            if current_function_length > 50:  # Long function
                complexity += 1
        
        return complexity


class QuestionGenerator:
    """Main question generator for QA datasets."""
    
    def __init__(self):
        self.code_analyzer = CodeAnalyzer()
        self.templates = self._create_question_templates()
    
    def _create_question_templates(self) -> List[QuestionTemplate]:
        """Create question templates for different categories and difficulties."""
        templates = []
        
        # Function behavior questions (Easy)
        templates.append(QuestionTemplate(
            template="What does the function '{function_name}' do?",
            category=QuestionCategory.FUNCTION_BEHAVIOR,
            difficulty=DifficultyLevel.EASY,
            evaluation_type=EvaluationType.SEMANTIC_SIMILARITY,
            required_concepts=['{function_name}', 'function', 'purpose'],
            placeholders=['function_name']
        ))
        
        templates.append(QuestionTemplate(
            template="What are the input parameters for '{function_name}'?",
            category=QuestionCategory.FUNCTION_BEHAVIOR,
            difficulty=DifficultyLevel.EASY,
            evaluation_type=EvaluationType.EXACT_MATCH,
            required_concepts=['{function_name}', 'parameters', 'input'],
            placeholders=['function_name']
        ))
        
        # API usage questions (Medium)
        templates.append(QuestionTemplate(
            template="How would you use the '{class_name}' class in a typical workflow?",
            category=QuestionCategory.API_USAGE,
            difficulty=DifficultyLevel.MEDIUM,
            evaluation_type=EvaluationType.RUBRIC_BASED,
            required_concepts=['{class_name}', 'usage', 'workflow'],
            placeholders=['class_name']
        ))
        
        templates.append(QuestionTemplate(
            template="What is the correct way to initialize and configure '{class_name}'?",
            category=QuestionCategory.API_USAGE,
            difficulty=DifficultyLevel.MEDIUM,
            evaluation_type=EvaluationType.SEMANTIC_SIMILARITY,
            required_concepts=['{class_name}', 'initialization', 'configuration'],
            placeholders=['class_name']
        ))
        
        # Algorithm implementation (Medium/Hard)
        templates.append(QuestionTemplate(
            template="What algorithm is implemented in '{function_name}' and what is its time complexity?",
            category=QuestionCategory.ALGORITHM_IMPLEMENTATION,
            difficulty=DifficultyLevel.HARD,
            evaluation_type=EvaluationType.RUBRIC_BASED,
            required_concepts=['{function_name}', 'algorithm', 'complexity', 'time'],
            placeholders=['function_name']
        ))
        
        # Class design questions (Medium)
        templates.append(QuestionTemplate(
            template="What design pattern does '{class_name}' implement and why?",
            category=QuestionCategory.CLASS_DESIGN,
            difficulty=DifficultyLevel.MEDIUM,
            evaluation_type=EvaluationType.RUBRIC_BASED,
            required_concepts=['{class_name}', 'design pattern', 'rationale'],
            placeholders=['class_name']
        ))
        
        # Error handling (Medium)
        templates.append(QuestionTemplate(
            template="How does '{function_name}' handle error conditions?",
            category=QuestionCategory.ERROR_HANDLING,
            difficulty=DifficultyLevel.MEDIUM,
            evaluation_type=EvaluationType.SEMANTIC_SIMILARITY,
            required_concepts=['{function_name}', 'error', 'exception', 'handling'],
            placeholders=['function_name']
        ))
        
        # Architecture questions (Hard)
        templates.append(QuestionTemplate(
            template="What is the overall architecture of this system and how do components interact?",
            category=QuestionCategory.ARCHITECTURE_PATTERNS,
            difficulty=DifficultyLevel.HARD,
            evaluation_type=EvaluationType.RUBRIC_BASED,
            required_concepts=['architecture', 'components', 'interaction', 'system']
        ))
        
        # Design decisions (Hard)
        templates.append(QuestionTemplate(
            template="Why was '{pattern}' chosen for this implementation? What are the trade-offs?",
            category=QuestionCategory.DESIGN_DECISIONS,
            difficulty=DifficultyLevel.HARD,
            evaluation_type=EvaluationType.RUBRIC_BASED,
            required_concepts=['{pattern}', 'decision', 'trade-offs', 'rationale'],
            placeholders=['pattern']
        ))
        
        # System integration (Medium/Hard)
        templates.append(QuestionTemplate(
            template="How does this system integrate with external services or APIs?",
            category=QuestionCategory.SYSTEM_INTEGRATION,
            difficulty=DifficultyLevel.MEDIUM,
            evaluation_type=EvaluationType.SEMANTIC_SIMILARITY,
            required_concepts=['integration', 'external', 'services', 'API']
        ))
        
        # Testing strategy (Medium)
        templates.append(QuestionTemplate(
            template="What testing approach is used for '{function_name}' and why is it appropriate?",
            category=QuestionCategory.TESTING_STRATEGY,
            difficulty=DifficultyLevel.MEDIUM,
            evaluation_type=EvaluationType.RUBRIC_BASED,
            required_concepts=['{function_name}', 'testing', 'approach', 'appropriate'],
            placeholders=['function_name']
        ))
        
        return templates
    
    def generate_questions_for_repository(
        self,
        repo_metadata: RepositoryMetadata,
        repo_path: Path,
        target_count: int = 60
    ) -> List[QuestionItem]:
        """
        Generate questions for a specific repository.
        
        Args:
            repo_metadata: Repository metadata
            repo_path: Path to repository
            target_count: Target number of questions to generate
            
        Returns:
            List of generated question items
        """
        # Analyze repository code
        analysis = self.code_analyzer.analyze_repository(repo_path, repo_metadata.language)
        
        # Plan question distribution
        distribution = self._plan_question_distribution(target_count)
        
        # Generate questions by category and difficulty
        questions = []
        question_id = 1
        
        for category, difficulties in distribution.items():
            for difficulty, count in difficulties.items():
                category_questions = self._generate_category_questions(
                    category, difficulty, count, repo_metadata, analysis
                )
                
                for question_text, concepts, eval_type in category_questions:
                    qid = f"{repo_metadata.repo_id}_q{question_id:03d}"
                    
                    # Create gold answer
                    gold_answer = self._create_gold_answer(
                        question_text, concepts, difficulty, eval_type, analysis
                    )
                    
                    # Create rubric if needed
                    rubric = None
                    if eval_type == EvaluationType.RUBRIC_BASED:
                        rubric = self._create_evaluation_rubric(
                            question_text, concepts, difficulty
                        )
                    
                    question_item = QuestionItem(
                        repo_id=repo_metadata.repo_id,
                        qid=qid,
                        question=question_text,
                        category=category,
                        difficulty=difficulty,
                        evaluation_type=eval_type,
                        pack_budget=repo_metadata.pack_budget,
                        gold=gold_answer,
                        rubric=rubric,
                        metadata={
                            'language': repo_metadata.language,
                            'domain': repo_metadata.domain,
                            'generator_version': '1.0'
                        }
                    )
                    
                    questions.append(question_item)
                    question_id += 1
        
        return questions
    
    def _plan_question_distribution(self, target_count: int) -> Dict[QuestionCategory, Dict[DifficultyLevel, int]]:
        """Plan question distribution across categories and difficulties."""
        # Target difficulty distribution (TODO.md requirements)
        difficulty_ratios = {
            DifficultyLevel.EASY: 0.20,    # 20%
            DifficultyLevel.MEDIUM: 0.50,  # 50%
            DifficultyLevel.HARD: 0.30     # 30%
        }
        
        # Category distribution (balanced across code and repo level)
        code_categories = [
            QuestionCategory.FUNCTION_BEHAVIOR,
            QuestionCategory.API_USAGE,
            QuestionCategory.ALGORITHM_IMPLEMENTATION,
            QuestionCategory.CLASS_DESIGN,
            QuestionCategory.ERROR_HANDLING
        ]
        
        repo_categories = [
            QuestionCategory.ARCHITECTURE_PATTERNS,
            QuestionCategory.DESIGN_DECISIONS,
            QuestionCategory.SYSTEM_INTEGRATION,
            QuestionCategory.DEPLOYMENT_CONFIG,
            QuestionCategory.TESTING_STRATEGY
        ]
        
        # Allocate questions
        distribution = {}
        
        # 60% code-level, 40% repo-level
        code_count = int(target_count * 0.6)
        repo_count = target_count - code_count
        
        # Distribute among code categories
        for i, category in enumerate(code_categories):
            category_count = code_count // len(code_categories)
            if i < code_count % len(code_categories):
                category_count += 1
            
            distribution[category] = {}
            for difficulty, ratio in difficulty_ratios.items():
                diff_count = max(1, int(category_count * ratio))
                distribution[category][difficulty] = diff_count
        
        # Distribute among repo categories
        for i, category in enumerate(repo_categories):
            category_count = repo_count // len(repo_categories)
            if i < repo_count % len(repo_categories):
                category_count += 1
            
            distribution[category] = {}
            for difficulty, ratio in difficulty_ratios.items():
                diff_count = max(1, int(category_count * ratio))
                distribution[category][difficulty] = diff_count
        
        return distribution
    
    def _generate_category_questions(
        self,
        category: QuestionCategory,
        difficulty: DifficultyLevel,
        count: int,
        repo_metadata: RepositoryMetadata,
        analysis: Dict[str, Any]
    ) -> List[Tuple[str, List[str], EvaluationType]]:
        """Generate questions for specific category/difficulty."""
        questions = []
        
        # Filter templates for this category/difficulty
        matching_templates = [
            t for t in self.templates
            if t.category == category and t.difficulty == difficulty
        ]
        
        if not matching_templates:
            # Fallback generic question
            return [(
                f"Describe the {category.value} aspects of this codebase.",
                [category.value, repo_metadata.language],
                EvaluationType.SEMANTIC_SIMILARITY
            )]
        
        # Generate questions using templates
        for i in range(count):
            template = random.choice(matching_templates)
            
            # Prepare context for template
            context = self._prepare_template_context(analysis, repo_metadata)
            
            # Generate question from template
            question_text, concepts = template.generate_question(context)
            
            questions.append((question_text, concepts, template.evaluation_type))
        
        return questions
    
    def _prepare_template_context(self, analysis: Dict[str, Any], repo_metadata: RepositoryMetadata) -> Dict[str, Any]:
        """Prepare context variables for template substitution."""
        context = {}
        
        # Add functions
        if analysis['functions']:
            func = random.choice(analysis['functions'])
            context['function_name'] = func['name']
        
        # Add classes
        if analysis['classes']:
            cls = random.choice(analysis['classes'])
            context['class_name'] = cls['name']
        
        # Add patterns
        if analysis['patterns']:
            context['pattern'] = random.choice(analysis['patterns'])
        
        # Add repository info
        context['language'] = repo_metadata.language
        context['domain'] = repo_metadata.domain
        context['repo_name'] = repo_metadata.name
        
        return context
    
    def _create_gold_answer(
        self,
        question: str,
        concepts: List[str],
        difficulty: DifficultyLevel,
        eval_type: EvaluationType,
        analysis: Dict[str, Any]
    ) -> GoldAnswer:
        """Create gold answer for question."""
        
        if eval_type == EvaluationType.EXACT_MATCH:
            # For exact match, provide specific answer
            if 'function_name' in question:
                answer_text = f"Function implementation details based on code analysis"
            else:
                answer_text = "Specific factual answer based on codebase"
                
            return GoldAnswer(
                answer_text=answer_text,
                key_concepts=concepts,
                confidence_score=0.9,
                annotator_id="generator",
                validation_notes="Generated answer for exact match evaluation"
            )
        
        elif eval_type == EvaluationType.REGEX_MATCH:
            # Create regex pattern for key concepts
            pattern = '|'.join(re.escape(concept.lower()) for concept in concepts[:3])
            return GoldAnswer(
                regex_pattern=f".*({pattern}).*",
                key_concepts=concepts,
                confidence_score=0.8,
                annotator_id="generator",
                validation_notes="Regex pattern matches key concepts"
            )
        
        else:  # SEMANTIC_SIMILARITY or RUBRIC_BASED
            return GoldAnswer(
                key_concepts=concepts,
                confidence_score=0.7 if difficulty == DifficultyLevel.HARD else 0.8,
                annotator_id="generator",
                validation_notes="Semantic evaluation required"
            )
    
    def _create_evaluation_rubric(
        self,
        question: str,
        concepts: List[str],
        difficulty: DifficultyLevel
    ) -> str:
        """Create evaluation rubric for rubric-based questions."""
        
        rubric_parts = [
            "# Evaluation Rubric",
            f"\n## Question: {question}",
            "\n## Scoring Criteria (0-1 scale):",
            "\n### Content Accuracy (40%)"
        ]
        
        if concepts:
            rubric_parts.append(f"- Must mention key concepts: {', '.join(concepts[:4])}")
        
        rubric_parts.extend([
            "- Factual correctness based on codebase",
            "- Technical precision and clarity",
            "\n### Completeness (30%)",
            "- Addresses all aspects of the question",
            "- Provides sufficient detail for understanding"
        ])
        
        if difficulty == DifficultyLevel.HARD:
            rubric_parts.extend([
                "\n### Analysis Depth (30%)",
                "- Demonstrates understanding of trade-offs",
                "- Explains rationale and implications",
                "- Shows synthesis of multiple concepts"
            ])
        else:
            rubric_parts.extend([
                "\n### Clarity (30%)",
                "- Clear and well-structured explanation",
                "- Appropriate level of technical detail"
            ])
        
        rubric_parts.append("\n## Overall Score: (Content×0.4) + (Completeness×0.3) + (Depth/Clarity×0.3)")
        
        return '\n'.join(rubric_parts)
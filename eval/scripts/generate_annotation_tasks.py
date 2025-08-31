#!/usr/bin/env python3
"""
Annotation Task Generation System
FastPath V5 Ground-Truth Protocol - ICSE 2025 Submission

This script generates structured annotation tasks for human annotators using
PR-modified file signals and repository context. Implements the systematic 
task design methodology specified in the ground-truth protocol.

Key Features:
- Automated task generation from PR signals
- Balanced difficulty distribution
- Bias mitigation through randomization
- Comprehensive task metadata for audit trail
- ICSE-compliant task design standards
"""

import json
import random
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import argparse
import logging
from collections import defaultdict, Counter

import numpy as np


@dataclass
class AnnotationTask:
    """Structured representation of an annotation task."""
    task_id: str
    task_batch: str
    repository: str
    file_path: str
    task_type: str  # feature_implementation, bug_fix, refactoring
    scenario_description: str
    specific_requirements: List[str]
    file_preview: str
    context_files: List[str]
    difficulty_level: str  # simple, medium, complex
    expected_annotation_time_minutes: int
    pr_signal_score: float
    pr_context: Dict[str, Any]
    task_metadata: Dict[str, Any]


@dataclass 
class TaskBatch:
    """Collection of annotation tasks for systematic distribution."""
    batch_id: str
    repository: str
    scenario_theme: str
    task_count: int
    difficulty_distribution: Dict[str, int]
    estimated_total_time_minutes: int
    tasks: List[AnnotationTask]
    generation_metadata: Dict[str, Any]


class AnnotationTaskGenerator:
    """
    Generate systematic annotation tasks for ground-truth dataset creation.
    
    Implements academic-grade task design with bias mitigation, balanced
    complexity distribution, and comprehensive metadata tracking for
    reproducibility and audit trail requirements.
    """
    
    def __init__(self, output_dir: Path, seed: int = 42):
        """Initialize task generator with reproducible settings."""
        self.output_dir = Path(output_dir)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Task generation parameters
        self.difficulty_weights = {
            'simple': 0.30,    # 30% simple tasks
            'medium': 0.45,    # 45% medium tasks  
            'complex': 0.25    # 25% complex tasks
        }
        
        self.task_time_estimates = {
            'simple': (3, 8),     # 3-8 minutes
            'medium': (8, 15),    # 8-15 minutes
            'complex': (15, 30)   # 15-30 minutes
        }
        
        # Scenario templates for different task types
        self.scenario_templates = self.load_scenario_templates()
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self) -> None:
        """Setup comprehensive logging for audit trail."""
        log_file = self.output_dir / 'task_generation_audit.log'
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
        
    def load_scenario_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined scenario templates for task generation."""
        return {
            'web_app_auth': {
                'theme': 'User Authentication System',
                'description': 'Implement comprehensive user authentication with login/logout functionality',
                'requirements': [
                    'User login form with email/password validation',
                    'Session management and persistence', 
                    'Password reset workflow',
                    'User profile editing interface',
                    'Authentication middleware for protected routes'
                ],
                'complexity_factors': ['async_operations', 'state_management', 'error_handling'],
                'applicable_languages': ['typescript', 'javascript', 'python']
            },
            'api_crud_operations': {
                'theme': 'REST API CRUD Implementation', 
                'description': 'Create complete CRUD operations for resource management',
                'requirements': [
                    'GET endpoint for resource listing with pagination',
                    'POST endpoint for resource creation with validation',
                    'PUT/PATCH endpoint for resource updates',
                    'DELETE endpoint with cascade handling',
                    'Error responses following RFC 7807 standard'
                ],
                'complexity_factors': ['database_operations', 'validation_logic', 'error_handling'],
                'applicable_languages': ['python', 'typescript', 'java', 'go']
            },
            'data_visualization': {
                'theme': 'Interactive Dashboard Components',
                'description': 'Build responsive dashboard with data visualization',
                'requirements': [
                    'Chart components for different data types',
                    'Real-time data updates via WebSocket',
                    'Filtering and search functionality',
                    'Export capabilities (PDF/CSV)',
                    'Mobile-responsive design'
                ],
                'complexity_factors': ['component_architecture', 'real_time_updates', 'responsive_design'],
                'applicable_languages': ['typescript', 'javascript']
            },
            'microservice_integration': {
                'theme': 'Microservice Communication Layer',
                'description': 'Implement service-to-service communication with resilience patterns',
                'requirements': [
                    'Service discovery and registration',
                    'Circuit breaker pattern implementation',
                    'Distributed tracing integration',
                    'Health check endpoints',
                    'API versioning and backward compatibility'
                ],
                'complexity_factors': ['distributed_systems', 'resilience_patterns', 'monitoring'],
                'applicable_languages': ['go', 'java', 'python']
            },
            'search_functionality': {
                'theme': 'Advanced Search and Filtering',
                'description': 'Implement full-text search with faceted filtering',
                'requirements': [
                    'Search query parsing and validation',
                    'Full-text search with ranking',
                    'Faceted filters for multiple dimensions',
                    'Auto-complete suggestions',
                    'Search result highlighting'
                ],
                'complexity_factors': ['search_algorithms', 'query_optimization', 'user_experience'],
                'applicable_languages': ['python', 'typescript', 'java']
            },
            'mobile_offline_sync': {
                'theme': 'Offline-First Mobile Features',
                'description': 'Implement offline data synchronization for mobile app',
                'requirements': [
                    'Local data storage with SQLite',
                    'Conflict resolution for concurrent edits',
                    'Background synchronization',
                    'Network state awareness',
                    'Progressive data loading'
                ],
                'complexity_factors': ['offline_storage', 'sync_algorithms', 'mobile_constraints'],
                'applicable_languages': ['typescript', 'javascript']
            }
        }
        
    def load_pr_signals(self, pr_signals_file: Path) -> List[Dict[str, Any]]:
        """Load PR-based relevance signals for task generation."""
        self.logger.info(f"Loading PR signals from {pr_signals_file}")
        
        with open(pr_signals_file, 'r') as f:
            data = json.load(f)
            
        signals = data.get('relevance_signals', [])
        self.logger.info(f"Loaded {len(signals)} PR relevance signals")
        return signals
        
    def select_scenario_for_repository(self, repo_context: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate scenario based on repository characteristics."""
        primary_language = repo_context.get('language_primary', '').lower()
        repo_size = repo_context.get('file_count', 0)
        
        # Filter scenarios by language compatibility
        applicable_scenarios = {
            name: template for name, template in self.scenario_templates.items()
            if primary_language in template['applicable_languages']
        }
        
        if not applicable_scenarios:
            # Fallback to language-agnostic scenarios
            applicable_scenarios = {
                name: template for name, template in self.scenario_templates.items()
                if 'typescript' in template['applicable_languages']  # Most general
            }
            
        # Prefer scenarios based on repository size/complexity
        if repo_size > 5000:  # Large repository
            preferred = ['microservice_integration', 'api_crud_operations']
        elif repo_size > 1000:  # Medium repository  
            preferred = ['web_app_auth', 'data_visualization', 'search_functionality']
        else:  # Small repository
            preferred = ['web_app_auth', 'api_crud_operations']
            
        # Select scenario with preference
        for pref in preferred:
            if pref in applicable_scenarios:
                return applicable_scenarios[pref]
                
        # Random selection if no preferred match
        return random.choice(list(applicable_scenarios.values()))
        
    def determine_file_difficulty(self, pr_signal: Dict[str, Any]) -> str:
        """Determine annotation difficulty based on PR signal characteristics."""
        relevance_score = pr_signal.get('relevance_score', 0.0)
        semantic_role = pr_signal.get('semantic_role', '')
        change_density = pr_signal.get('change_density', 0.0)
        dependency_impact = pr_signal.get('dependency_impact', 0.0)
        
        # Complexity scoring
        complexity_score = 0.0
        
        # Semantic role complexity
        role_complexity = {
            'core_logic': 0.8,
            'configuration': 0.4,
            'test': 0.3,
            'build_deployment': 0.5,
            'documentation': 0.2,
            'assets': 0.1
        }
        complexity_score += role_complexity.get(semantic_role, 0.5)
        
        # Change characteristics
        complexity_score += min(change_density * 0.5, 0.3)
        complexity_score += min(dependency_impact * 0.4, 0.3)
        complexity_score += min(relevance_score * 0.3, 0.2)
        
        # Classify based on composite score
        if complexity_score >= 0.7:
            return 'complex'
        elif complexity_score >= 0.4:
            return 'medium'
        else:
            return 'simple'
            
    def generate_file_preview(self, file_path: str, pr_context: Dict[str, Any]) -> str:
        """Generate realistic file preview for annotation context."""
        file_ext = Path(file_path).suffix.lower()
        
        # Generate preview based on file type
        if file_ext in ['.ts', '.tsx']:
            return self.generate_typescript_preview(file_path, pr_context)
        elif file_ext in ['.py']:
            return self.generate_python_preview(file_path, pr_context)
        elif file_ext in ['.java']:
            return self.generate_java_preview(file_path, pr_context)
        elif file_ext in ['.go']:
            return self.generate_go_preview(file_path, pr_context)
        else:
            return self.generate_generic_preview(file_path, pr_context)
            
    def generate_typescript_preview(self, file_path: str, pr_context: Dict[str, Any]) -> str:
        """Generate TypeScript file preview."""
        filename = Path(file_path).stem
        
        if 'component' in file_path.lower() or filename.endswith('Component'):
            return f"""// {filename}.tsx
import React, {{ useState, useEffect }} from 'react';
import {{ {filename}Props }} from './types';

export const {filename}: React.FC<{filename}Props> = ({{ 
  // Props extracted from context
}}) => {{
  const [state, setState] = useState(initialState);
  
  useEffect(() => {{
    // Component lifecycle logic
  }}, [dependencies]);
  
  const handleAction = () => {{
    // Event handling logic
  }};
  
  return (
    <div className="{filename.lower()}">
      {{/* Component JSX */}}
    </div>
  );
}};"""
        else:
            return f"""// {filename}.ts
export interface {filename}Config {{
  // Configuration interface
}}

export class {filename} {{
  private readonly config: {filename}Config;
  
  constructor(config: {filename}Config) {{
    this.config = config;
  }}
  
  public async execute(): Promise<void> {{
    // Implementation logic
  }}
}}"""

    def generate_python_preview(self, file_path: str, pr_context: Dict[str, Any]) -> str:
        """Generate Python file preview."""
        filename = Path(file_path).stem
        
        return f'''# {filename}.py
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class {filename.title()}Config:
    """Configuration for {filename} functionality."""
    # Configuration fields
    pass

class {filename.title()}:
    """Core {filename} implementation."""
    
    def __init__(self, config: {filename.title()}Config):
        self.config = config
        
    async def process(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process data according to configuration."""
        # Implementation logic
        return None'''

    def generate_java_preview(self, file_path: str, pr_context: Dict[str, Any]) -> str:
        """Generate Java file preview."""
        filename = Path(file_path).stem
        
        return f'''// {filename}.java
package com.example.service;

import java.util.Optional;
import java.util.concurrent.CompletableFuture;

public class {filename} {{
    private final {filename}Config config;
    
    public {filename}({filename}Config config) {{
        this.config = config;
    }}
    
    public CompletableFuture<Optional<Result>> execute() {{
        // Implementation logic
        return CompletableFuture.completedFuture(Optional.empty());
    }}
}}'''

    def generate_go_preview(self, file_path: str, pr_context: Dict[str, Any]) -> str:
        """Generate Go file preview."""
        filename = Path(file_path).stem
        
        return f'''// {filename}.go
package main

import (
    "context"
    "fmt"
)

type {filename.title()}Config struct {{
    // Configuration fields
}}

type {filename.title()} struct {{
    config *{filename.title()}Config
}}

func New{filename.title()}(config *{filename.title()}Config) *{filename.title()} {{
    return &{filename.title()}{{config: config}}
}}

func (s *{filename.title()}) Execute(ctx context.Context) error {{
    // Implementation logic
    return nil
}}'''

    def generate_generic_preview(self, file_path: str, pr_context: Dict[str, Any]) -> str:
        """Generate generic file preview."""
        return f"""# {Path(file_path).name}

This file contains implementation related to:
- File modifications from PR #{pr_context.get('pr_number', 'N/A')}
- Change type: {pr_context.get('change_type', 'modified')}
- [Preview content would be extracted from actual file]

Key areas likely requiring attention:
- Core functionality implementation
- Error handling and validation
- Integration with existing systems
"""

    def identify_context_files(self, target_file: str, pr_signals: List[Dict[str, Any]]) -> List[str]:
        """Identify related context files for annotation task."""
        target_dir = str(Path(target_file).parent)
        target_name = Path(target_file).stem
        
        context_files = []
        
        # Find files in same directory
        same_dir_files = [
            signal['file_path'] for signal in pr_signals
            if str(Path(signal['file_path']).parent) == target_dir
            and signal['file_path'] != target_file
        ]
        context_files.extend(same_dir_files[:3])  # Max 3 same-directory files
        
        # Find related files by name patterns
        related_files = [
            signal['file_path'] for signal in pr_signals
            if target_name.lower() in Path(signal['file_path']).stem.lower()
            and signal['file_path'] != target_file
            and signal['file_path'] not in context_files
        ]
        context_files.extend(related_files[:2])  # Max 2 name-related files
        
        # Find high-relevance files as additional context
        high_relevance_files = [
            signal['file_path'] for signal in pr_signals
            if signal.get('relevance_score', 0) >= 0.7
            and signal['file_path'] != target_file
            and signal['file_path'] not in context_files
        ]
        context_files.extend(high_relevance_files[:2])  # Max 2 high-relevance files
        
        return context_files
        
    def calculate_annotation_time(self, difficulty: str, context_file_count: int) -> int:
        """Calculate estimated annotation time based on task characteristics."""
        base_min, base_max = self.task_time_estimates[difficulty]
        base_time = random.randint(base_min, base_max)
        
        # Add time for context files
        context_time = context_file_count * random.randint(1, 3)
        
        return base_time + context_time
        
    def generate_task_from_signal(self, pr_signal: Dict[str, Any], 
                                scenario: Dict[str, Any],
                                pr_signals: List[Dict[str, Any]],
                                task_batch_id: str,
                                repository: str) -> AnnotationTask:
        """Generate a complete annotation task from PR signal."""
        
        file_path = pr_signal['file_path']
        difficulty = self.determine_file_difficulty(pr_signal)
        context_files = self.identify_context_files(file_path, pr_signals)
        
        # Generate unique task ID
        task_id = hashlib.sha256(
            f"{task_batch_id}_{file_path}_{pr_signal.get('pr_context', {}).get('pr_number', 0)}"
            .encode()
        ).hexdigest()[:12]
        
        # Create task
        task = AnnotationTask(
            task_id=task_id,
            task_batch=task_batch_id,
            repository=repository,
            file_path=file_path,
            task_type=scenario['theme'].lower().replace(' ', '_'),
            scenario_description=scenario['description'],
            specific_requirements=scenario['requirements'],
            file_preview=self.generate_file_preview(file_path, pr_signal.get('pr_context', {})),
            context_files=context_files,
            difficulty_level=difficulty,
            expected_annotation_time_minutes=self.calculate_annotation_time(difficulty, len(context_files)),
            pr_signal_score=pr_signal.get('relevance_score', 0.0),
            pr_context=pr_signal.get('pr_context', {}),
            task_metadata={
                'generation_timestamp': datetime.now(timezone.utc).isoformat(),
                'generator_version': '1.0.0',
                'methodology_version': 'ground_truth_v1.0',
                'seed_used': self.seed,
                'difficulty_factors': {
                    'semantic_role': pr_signal.get('semantic_role'),
                    'change_density': pr_signal.get('change_density'),
                    'dependency_impact': pr_signal.get('dependency_impact')
                }
            }
        )
        
        return task
        
    def balance_difficulty_distribution(self, tasks: List[AnnotationTask]) -> List[AnnotationTask]:
        """Balance task difficulty distribution according to target weights."""
        
        # Group by difficulty
        by_difficulty = defaultdict(list)
        for task in tasks:
            by_difficulty[task.difficulty_level].append(task)
            
        # Calculate target counts
        total_tasks = len(tasks)
        target_counts = {
            difficulty: int(total_tasks * weight)
            for difficulty, weight in self.difficulty_weights.items()
        }
        
        # Balance by sampling
        balanced_tasks = []
        for difficulty, target_count in target_counts.items():
            available = by_difficulty[difficulty]
            if len(available) >= target_count:
                # Random sample without replacement
                sampled = random.sample(available, target_count)
            else:
                # Use all available
                sampled = available
                
            balanced_tasks.extend(sampled)
            
        # If we need more tasks, add from remaining pool
        remaining_tasks = [
            task for task in tasks if task not in balanced_tasks
        ]
        
        additional_needed = total_tasks - len(balanced_tasks)
        if additional_needed > 0 and remaining_tasks:
            additional = random.sample(
                remaining_tasks, 
                min(additional_needed, len(remaining_tasks))
            )
            balanced_tasks.extend(additional)
            
        # Shuffle for random presentation order
        random.shuffle(balanced_tasks)
        
        return balanced_tasks
        
    def generate_batch_from_repository(self, pr_signals_file: Path,
                                     repo_context: Dict[str, Any],
                                     max_tasks_per_batch: int = 15) -> TaskBatch:
        """Generate a complete task batch for a repository."""
        
        # Load PR signals
        pr_signals = self.load_pr_signals(pr_signals_file)
        
        if not pr_signals:
            raise ValueError(f"No PR signals found in {pr_signals_file}")
            
        # Select scenario
        scenario = self.select_scenario_for_repository(repo_context)
        
        # Generate batch ID
        repo_name = repo_context.get('name', 'unknown_repo')
        batch_id = f"{repo_name}_{scenario['theme'].lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"
        
        # Filter signals for task generation (focus on higher relevance)
        suitable_signals = [
            signal for signal in pr_signals
            if signal.get('relevance_score', 0.0) >= 0.2  # Minimum relevance threshold
        ]
        
        # Limit to max tasks per batch
        if len(suitable_signals) > max_tasks_per_batch:
            # Stratified sampling by relevance score
            suitable_signals.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
            suitable_signals = suitable_signals[:max_tasks_per_batch]
            
        # Generate tasks
        tasks = []
        for signal in suitable_signals:
            task = self.generate_task_from_signal(
                signal, scenario, pr_signals, batch_id, repo_name
            )
            tasks.append(task)
            
        # Balance difficulty distribution
        tasks = self.balance_difficulty_distribution(tasks)
        
        # Calculate batch statistics
        difficulty_distribution = Counter(task.difficulty_level for task in tasks)
        total_time = sum(task.expected_annotation_time_minutes for task in tasks)
        
        # Create batch
        batch = TaskBatch(
            batch_id=batch_id,
            repository=repo_name,
            scenario_theme=scenario['theme'],
            task_count=len(tasks),
            difficulty_distribution=dict(difficulty_distribution),
            estimated_total_time_minutes=total_time,
            tasks=tasks,
            generation_metadata={
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'generator_version': '1.0.0',
                'methodology_version': 'ground_truth_v1.0',
                'seed': self.seed,
                'pr_signals_file': str(pr_signals_file),
                'scenario_selected': scenario['theme'],
                'target_difficulty_weights': self.difficulty_weights,
                'actual_difficulty_distribution': dict(difficulty_distribution)
            }
        )
        
        return batch
        
    def save_task_batch(self, batch: TaskBatch, output_filename: str) -> Path:
        """Save task batch with comprehensive metadata."""
        
        # Prepare serializable batch data
        batch_data = {
            'batch_metadata': {
                'batch_id': batch.batch_id,
                'repository': batch.repository,
                'scenario_theme': batch.scenario_theme,
                'task_count': batch.task_count,
                'difficulty_distribution': batch.difficulty_distribution,
                'estimated_total_time_minutes': batch.estimated_total_time_minutes,
                'generation_metadata': batch.generation_metadata
            },
            'annotation_instructions': {
                'relevance_scale_definition': {
                    '5': 'Critical - File is essential for task completion',
                    '4': 'High - File likely requires modification or deep understanding', 
                    '3': 'Medium - File provides important context or may need minor changes',
                    '2': 'Low - File provides some context but unlikely to need changes',
                    '1': 'Minimal - File is largely irrelevant to the task'
                },
                'confidence_scale_definition': {
                    '5': 'Very confident in relevance assessment',
                    '4': 'Confident with minor uncertainty',
                    '3': 'Moderately confident',
                    '2': 'Low confidence, significant uncertainty',
                    '1': 'Very low confidence, mostly guessing'
                },
                'annotation_guidelines': [
                    'Read the scenario description and requirements carefully',
                    'Review the file preview and context files',
                    'Consider how the file would be involved in implementing the scenario',
                    'Rate relevance based on likelihood of modification or importance for understanding',
                    'Rate confidence based on how certain you are about the relevance assessment',
                    'Provide clear reasoning for your relevance rating'
                ]
            },
            'tasks': [asdict(task) for task in batch.tasks]
        }
        
        # Save to JSON
        output_path = self.output_dir / output_filename
        with open(output_path, 'w') as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Task batch saved to {output_path}")
        return output_path
        
    def validate_batch_quality(self, batch: TaskBatch) -> Dict[str, Any]:
        """Validate batch quality against academic standards."""
        
        quality_report = {
            'task_count_adequate': batch.task_count >= 10,
            'difficulty_balanced': all(
                count >= 2 for count in batch.difficulty_distribution.values()
            ),
            'time_estimate_reasonable': (
                30 <= batch.estimated_total_time_minutes <= 180  # 30min - 3hrs
            ),
            'context_files_provided': all(
                len(task.context_files) >= 1 for task in batch.tasks
            ),
            'preview_content_generated': all(
                len(task.file_preview) >= 50 for task in batch.tasks
            ),
            'pr_signals_linked': all(
                task.pr_signal_score > 0.0 for task in batch.tasks
            )
        }
        
        quality_report['overall_quality_score'] = sum(quality_report.values()) / len(quality_report)
        
        return quality_report


def main():
    """Main execution function for annotation task generation."""
    parser = argparse.ArgumentParser(
        description='Generate annotation tasks for ground-truth dataset creation'
    )
    parser.add_argument('--pr-signals-file', required=True, type=Path,
                       help='JSON file containing PR relevance signals')
    parser.add_argument('--repo-context-file', required=True, type=Path,
                       help='JSON file containing repository context')
    parser.add_argument('--output-dir', required=True, type=Path,
                       help='Output directory for task batch')
    parser.add_argument('--max-tasks', type=int, default=15,
                       help='Maximum tasks per batch')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = AnnotationTaskGenerator(
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    try:
        # Load repository context
        with open(args.repo_context_file, 'r') as f:
            repo_context = json.load(f)
            
        # Generate task batch
        batch = generator.generate_batch_from_repository(
            args.pr_signals_file,
            repo_context,
            args.max_tasks
        )
        
        # Validate batch quality
        quality_report = generator.validate_batch_quality(batch)
        
        # Save batch
        output_filename = f"annotation_tasks_{batch.repository}_{datetime.now().strftime('%Y%m%d')}.json"
        batch_path = generator.save_task_batch(batch, output_filename)
        
        # Print summary
        print("\n" + "="*60)
        print("ANNOTATION TASK GENERATION COMPLETE")
        print("="*60)
        print(f"📁 Repository: {batch.repository}")
        print(f"🎯 Scenario: {batch.scenario_theme}")
        print(f"📊 Task count: {batch.task_count}")
        print(f"⏱️  Estimated time: {batch.estimated_total_time_minutes} minutes")
        print(f"📈 Difficulty distribution: {batch.difficulty_distribution}")
        print(f"✅ Quality score: {quality_report['overall_quality_score']:.2f}")
        print(f"📁 Tasks saved to: {batch_path}")
        print("="*60)
        
    except Exception as e:
        generator.logger.error(f"Task generation failed: {e}")
        print(f"❌ Generation failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
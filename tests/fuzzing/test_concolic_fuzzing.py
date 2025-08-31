"""
Concolic (concrete + symbolic) testing for PackRepo boundary logic.

Implements concolic testing to systematically explore execution paths
in critical PackRepo algorithms, particularly around boundary conditions
in selection, chunking, and budget enforcement logic.
"""

from __future__ import annotations

import pytest
import ast
import random
import copy
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

from .fuzzer_engine import FuzzerEngine, PackRepoSelectorTarget, PackRepoChunkerTarget


class ConstraintType(Enum):
    """Types of symbolic constraints."""
    EQUALITY = "=="
    INEQUALITY_LT = "<"
    INEQUALITY_LE = "<="
    INEQUALITY_GT = ">"
    INEQUALITY_GE = ">="
    NOT_EQUAL = "!="


@dataclass
class SymbolicVariable:
    """Represents a symbolic variable in concolic execution."""
    name: str
    concrete_value: Any
    constraints: List['SymbolicConstraint'] = field(default_factory=list)
    
    def __str__(self):
        return f"{self.name}={self.concrete_value}"


@dataclass
class SymbolicConstraint:
    """Represents a constraint on symbolic variables."""
    variable: str
    constraint_type: ConstraintType
    value: Any
    
    def __str__(self):
        return f"{self.variable} {self.constraint_type.value} {self.value}"


@dataclass
class ExecutionPath:
    """Represents an execution path with constraints."""
    path_id: str
    constraints: List[SymbolicConstraint]
    concrete_inputs: Dict[str, Any]
    execution_result: Any = None
    error: Optional[Exception] = None
    
    def __str__(self):
        constraints_str = ", ".join(str(c) for c in self.constraints)
        return f"Path {self.path_id}: [{constraints_str}] -> {self.execution_result}"


class ConcolicExecutionEngine:
    """Concolic execution engine for systematic path exploration."""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed or 42
        random.seed(self.seed)
        
        self.explored_paths: List[ExecutionPath] = []
        self.symbolic_variables: Dict[str, SymbolicVariable] = {}
        self.current_constraints: List[SymbolicConstraint] = []
        self.path_counter = 0
        
    def reset(self):
        """Reset execution state."""
        self.current_constraints = []
        self.symbolic_variables = {}
    
    def create_symbolic_variable(self, name: str, concrete_value: Any) -> SymbolicVariable:
        """Create a symbolic variable with concrete value."""
        var = SymbolicVariable(name, concrete_value)
        self.symbolic_variables[name] = var
        return var
    
    def add_constraint(self, variable: str, constraint_type: ConstraintType, value: Any):
        """Add constraint to current path."""
        constraint = SymbolicConstraint(variable, constraint_type, value)
        self.current_constraints.append(constraint)
        
        # Update symbolic variable constraints
        if variable in self.symbolic_variables:
            self.symbolic_variables[variable].constraints.append(constraint)
    
    def explore_budget_boundary_paths(self, target_function: Callable) -> List[ExecutionPath]:
        """Explore execution paths around budget boundaries."""
        
        # Define budget boundary test cases
        budget_scenarios = [
            # Basic boundary cases
            {'budget': 100, 'cost': 50, 'scenario': 'under_budget'},
            {'budget': 100, 'cost': 100, 'scenario': 'exact_budget'}, 
            {'budget': 100, 'cost': 101, 'scenario': 'over_budget'},
            {'budget': 100, 'cost': 99, 'scenario': 'just_under'},
            {'budget': 100, 'cost': 102, 'scenario': 'just_over'},
            
            # Edge cases
            {'budget': 1, 'cost': 1, 'scenario': 'minimal_exact'},
            {'budget': 1, 'cost': 0, 'scenario': 'zero_cost'},
            {'budget': 0, 'cost': 1, 'scenario': 'zero_budget'},
            {'budget': 1000000, 'cost': 999999, 'scenario': 'large_just_under'},
            
            # Multiple items
            {'budget': 100, 'costs': [30, 30, 30, 30], 'scenario': 'multiple_over'},
            {'budget': 100, 'costs': [25, 25, 25, 25], 'scenario': 'multiple_exact'},
            {'budget': 100, 'costs': [20, 20, 20, 20], 'scenario': 'multiple_under'},
        ]
        
        paths = []
        
        for scenario_data in budget_scenarios:
            self.reset()
            self.path_counter += 1
            
            path_id = f"budget_path_{self.path_counter}"
            
            try:
                # Create symbolic variables for budget scenario
                budget = self.create_symbolic_variable('budget', scenario_data['budget'])
                
                if 'cost' in scenario_data:
                    # Single cost scenario
                    cost = self.create_symbolic_variable('cost', scenario_data['cost'])
                    
                    # Add boundary constraints
                    if scenario_data['cost'] < scenario_data['budget']:
                        self.add_constraint('cost', ConstraintType.INEQUALITY_LT, budget.concrete_value)
                    elif scenario_data['cost'] == scenario_data['budget']:
                        self.add_constraint('cost', ConstraintType.EQUALITY, budget.concrete_value)
                    else:
                        self.add_constraint('cost', ConstraintType.INEQUALITY_GT, budget.concrete_value)
                    
                    # Execute with concrete values
                    test_input = {
                        'chunks': [{'id': 'test_chunk', 'cost': cost.concrete_value, 'score': 1.0}],
                        'budget': budget.concrete_value
                    }
                    
                elif 'costs' in scenario_data:
                    # Multiple costs scenario
                    total_cost = sum(scenario_data['costs'])
                    total_cost_var = self.create_symbolic_variable('total_cost', total_cost)
                    
                    if total_cost < scenario_data['budget']:
                        self.add_constraint('total_cost', ConstraintType.INEQUALITY_LT, budget.concrete_value)
                    elif total_cost == scenario_data['budget']:
                        self.add_constraint('total_cost', ConstraintType.EQUALITY, budget.concrete_value)
                    else:
                        self.add_constraint('total_cost', ConstraintType.INEQUALITY_GT, budget.concrete_value)
                    
                    # Execute with multiple chunks
                    chunks = []
                    for i, cost in enumerate(scenario_data['costs']):
                        chunks.append({
                            'id': f'chunk_{i}',
                            'cost': cost,
                            'score': 1.0 / (i + 1)  # Decreasing scores
                        })
                    
                    test_input = {
                        'chunks': chunks,
                        'budget': budget.concrete_value
                    }
                
                # Execute target function
                result = target_function(test_input)
                
                # Create execution path
                path = ExecutionPath(
                    path_id=path_id,
                    constraints=copy.deepcopy(self.current_constraints),
                    concrete_inputs=test_input,
                    execution_result=result
                )
                
                paths.append(path)
                
            except Exception as e:
                # Record failed execution path
                path = ExecutionPath(
                    path_id=path_id,
                    constraints=copy.deepcopy(self.current_constraints),
                    concrete_inputs=scenario_data,
                    error=e
                )
                paths.append(path)
        
        self.explored_paths.extend(paths)
        return paths
    
    def explore_chunking_boundary_paths(self, target_function: Callable) -> List[ExecutionPath]:
        """Explore execution paths around chunking boundaries."""
        
        # Chunking boundary scenarios
        chunking_scenarios = [
            # Line boundary cases
            {
                'content': 'line1\nline2\nline3',
                'max_lines_per_chunk': 1,
                'scenario': 'single_line_chunks'
            },
            {
                'content': 'line1\nline2\nline3',
                'max_lines_per_chunk': 3,
                'scenario': 'exact_line_fit'
            },
            {
                'content': 'line1\nline2\nline3\nline4',
                'max_lines_per_chunk': 3,
                'scenario': 'line_overflow'
            },
            
            # Character boundary cases
            {
                'content': 'a' * 100,
                'max_chars_per_chunk': 100,
                'scenario': 'exact_char_fit'
            },
            {
                'content': 'a' * 101,
                'max_chars_per_chunk': 100,
                'scenario': 'char_overflow'
            },
            {
                'content': 'a' * 99,
                'max_chars_per_chunk': 100,
                'scenario': 'char_underflow'
            },
            
            # Empty and minimal cases
            {
                'content': '',
                'max_lines_per_chunk': 10,
                'scenario': 'empty_content'
            },
            {
                'content': 'single_line',
                'max_lines_per_chunk': 10,
                'scenario': 'single_line_under_limit'
            },
            
            # Unicode boundary cases
            {
                'content': '测试\n中文\n内容',
                'max_chars_per_chunk': 6,  # 2 chars per line
                'scenario': 'unicode_boundary'
            }
        ]
        
        paths = []
        
        for scenario_data in chunking_scenarios:
            self.reset()
            self.path_counter += 1
            
            path_id = f"chunking_path_{self.path_counter}"
            
            try:
                # Create symbolic variables
                content = self.create_symbolic_variable('content', scenario_data['content'])
                content_length = len(scenario_data['content'])
                content_lines = len(scenario_data['content'].split('\n'))
                
                # Add constraints based on chunking parameters
                if 'max_lines_per_chunk' in scenario_data:
                    max_lines = self.create_symbolic_variable('max_lines', scenario_data['max_lines_per_chunk'])
                    
                    if content_lines <= max_lines.concrete_value:
                        self.add_constraint('content_lines', ConstraintType.INEQUALITY_LE, max_lines.concrete_value)
                    else:
                        self.add_constraint('content_lines', ConstraintType.INEQUALITY_GT, max_lines.concrete_value)
                
                if 'max_chars_per_chunk' in scenario_data:
                    max_chars = self.create_symbolic_variable('max_chars', scenario_data['max_chars_per_chunk'])
                    
                    if content_length <= max_chars.concrete_value:
                        self.add_constraint('content_length', ConstraintType.INEQUALITY_LE, max_chars.concrete_value)
                    else:
                        self.add_constraint('content_length', ConstraintType.INEQUALITY_GT, max_chars.concrete_value)
                
                # Prepare input for chunker
                test_input = {
                    'content': content.concrete_value,
                    'file_path': f'{scenario_data["scenario"]}.py',
                    'config': {}
                }
                
                # Add chunking limits to config
                if 'max_lines_per_chunk' in scenario_data:
                    test_input['config']['max_lines_per_chunk'] = scenario_data['max_lines_per_chunk']
                
                if 'max_chars_per_chunk' in scenario_data:
                    test_input['config']['max_chars_per_chunk'] = scenario_data['max_chars_per_chunk']
                
                # Execute target function
                result = target_function(test_input)
                
                # Create execution path
                path = ExecutionPath(
                    path_id=path_id,
                    constraints=copy.deepcopy(self.current_constraints),
                    concrete_inputs=test_input,
                    execution_result=result
                )
                
                paths.append(path)
                
            except Exception as e:
                # Record failed execution path
                path = ExecutionPath(
                    path_id=path_id,
                    constraints=copy.deepcopy(self.current_constraints),
                    concrete_inputs=scenario_data,
                    error=e
                )
                paths.append(path)
        
        self.explored_paths.extend(paths)
        return paths
    
    def explore_selection_boundary_paths(self, target_function: Callable) -> List[ExecutionPath]:
        """Explore execution paths around selection algorithm boundaries."""
        
        # Selection boundary scenarios
        selection_scenarios = [
            # Score-based boundaries
            {
                'chunks': [
                    {'id': 'high', 'cost': 50, 'score': 1.0},
                    {'id': 'med', 'cost': 50, 'score': 0.5},
                    {'id': 'low', 'cost': 50, 'score': 0.1}
                ],
                'budget': 100,
                'scenario': 'score_boundary_two_items'
            },
            
            # Cost efficiency boundaries  
            {
                'chunks': [
                    {'id': 'efficient', 'cost': 10, 'score': 1.0},    # ratio = 0.1
                    {'id': 'medium', 'cost': 50, 'score': 1.0},      # ratio = 0.5
                    {'id': 'expensive', 'cost': 100, 'score': 1.0}   # ratio = 1.0
                ],
                'budget': 60,
                'scenario': 'cost_efficiency_boundary'
            },
            
            # Greedy vs optimal boundaries
            {
                'chunks': [
                    {'id': 'greedy_choice', 'cost': 80, 'score': 0.9},
                    {'id': 'optimal1', 'cost': 50, 'score': 0.6},
                    {'id': 'optimal2', 'cost': 50, 'score': 0.6}
                ],
                'budget': 100,
                'scenario': 'greedy_vs_optimal'
            },
            
            # Tie-breaking scenarios
            {
                'chunks': [
                    {'id': 'tie1', 'cost': 50, 'score': 0.5},
                    {'id': 'tie2', 'cost': 50, 'score': 0.5},
                    {'id': 'tie3', 'cost': 50, 'score': 0.5}
                ],
                'budget': 100,
                'scenario': 'tie_breaking'
            },
            
            # Submodular function boundaries
            {
                'chunks': [
                    {'id': 'base', 'cost': 30, 'score': 1.0},
                    {'id': 'redundant', 'cost': 30, 'score': 0.2},  # Low marginal utility
                    {'id': 'complement', 'cost': 30, 'score': 0.8}
                ],
                'budget': 90,
                'scenario': 'submodular_diminishing_returns'
            }
        ]
        
        paths = []
        
        for scenario_data in selection_scenarios:
            self.reset()
            self.path_counter += 1
            
            path_id = f"selection_path_{self.path_counter}"
            
            try:
                # Create symbolic variables for selection scenario
                budget = self.create_symbolic_variable('budget', scenario_data['budget'])
                num_chunks = len(scenario_data['chunks'])
                total_cost = sum(chunk['cost'] for chunk in scenario_data['chunks'])
                
                # Add constraints based on selection logic
                if total_cost <= budget.concrete_value:
                    self.add_constraint('total_cost', ConstraintType.INEQUALITY_LE, budget.concrete_value)
                else:
                    self.add_constraint('total_cost', ConstraintType.INEQUALITY_GT, budget.concrete_value)
                
                # Add constraints for chunk properties
                scores = [chunk['score'] for chunk in scenario_data['chunks']]
                costs = [chunk['cost'] for chunk in scenario_data['chunks']]
                
                max_score = max(scores)
                min_score = min(scores)
                max_cost = max(costs)
                min_cost = min(costs)
                
                if max_score != min_score:
                    # Score diversity - affects selection decisions
                    score_range = self.create_symbolic_variable('score_range', max_score - min_score)
                    self.add_constraint('score_range', ConstraintType.INEQUALITY_GT, 0)
                
                if max_cost != min_cost:
                    # Cost diversity - affects efficiency calculations
                    cost_range = self.create_symbolic_variable('cost_range', max_cost - min_cost)
                    self.add_constraint('cost_range', ConstraintType.INEQUALITY_GT, 0)
                
                # Execute target function
                test_input = {
                    'chunks': scenario_data['chunks'],
                    'budget': budget.concrete_value
                }
                
                result = target_function(test_input)
                
                # Create execution path
                path = ExecutionPath(
                    path_id=path_id,
                    constraints=copy.deepcopy(self.current_constraints),
                    concrete_inputs=test_input,
                    execution_result=result
                )
                
                paths.append(path)
                
            except Exception as e:
                # Record failed execution path
                path = ExecutionPath(
                    path_id=path_id,
                    constraints=copy.deepcopy(self.current_constraints),
                    concrete_inputs=scenario_data,
                    error=e
                )
                paths.append(path)
        
        self.explored_paths.extend(paths)
        return paths
    
    def generate_path_coverage_report(self) -> Dict[str, Any]:
        """Generate a report on path coverage achieved."""
        
        total_paths = len(self.explored_paths)
        successful_paths = [p for p in self.explored_paths if p.error is None]
        failed_paths = [p for p in self.explored_paths if p.error is not None]
        
        # Analyze constraint coverage
        constraint_types_seen = set()
        variable_names_seen = set()
        
        for path in self.explored_paths:
            for constraint in path.constraints:
                constraint_types_seen.add(constraint.constraint_type)
                variable_names_seen.add(constraint.variable)
        
        # Analyze path diversity
        unique_constraint_sets = set()
        for path in self.explored_paths:
            constraint_signature = tuple(sorted(str(c) for c in path.constraints))
            unique_constraint_sets.add(constraint_signature)
        
        report = {
            'total_paths_explored': total_paths,
            'successful_executions': len(successful_paths),
            'failed_executions': len(failed_paths),
            'success_rate': len(successful_paths) / total_paths if total_paths > 0 else 0,
            'constraint_types_covered': list(constraint_types_seen),
            'variables_analyzed': list(variable_names_seen),
            'unique_constraint_combinations': len(unique_constraint_sets),
            'path_diversity_score': len(unique_constraint_sets) / total_paths if total_paths > 0 else 0,
            'sample_successful_paths': [str(p) for p in successful_paths[:5]],
            'sample_failed_paths': [f"{p.path_id}: {p.error}" for p in failed_paths[:5]]
        }
        
        return report


class TestConcolicFuzzing:
    """Concolic testing for PackRepo boundary logic."""
    
    @pytest.fixture
    def concolic_engine(self):
        """Create concolic execution engine."""
        return ConcolicExecutionEngine(seed=54321)
    
    def test_budget_boundary_concolic_exploration(self, concolic_engine):
        """Test concolic exploration of budget boundaries."""
        
        # Create target function wrapper
        selector_target = PackRepoSelectorTarget()
        
        def budget_target_function(test_input):
            return selector_target.execute(test_input)
        
        # Explore budget boundary paths
        paths = concolic_engine.explore_budget_boundary_paths(budget_target_function)
        
        # Analyze path exploration results
        assert len(paths) > 0, "Should explore multiple budget boundary paths"
        
        successful_paths = [p for p in paths if p.error is None]
        failed_paths = [p for p in paths if p.error is not None]
        
        success_rate = len(successful_paths) / len(paths)
        
        print(f"Budget boundary concolic exploration results:")
        print(f"  Total paths explored: {len(paths)}")
        print(f"  Successful paths: {len(successful_paths)}")
        print(f"  Failed paths: {len(failed_paths)}")
        print(f"  Success rate: {success_rate:.2%}")
        
        # Show sample constraint exploration
        if successful_paths:
            print(f"  Sample successful constraint explorations:")
            for path in successful_paths[:3]:
                constraints_str = ", ".join(str(c) for c in path.constraints)
                print(f"    {path.path_id}: [{constraints_str}]")
        
        if failed_paths:
            print(f"  Sample failed explorations:")
            for path in failed_paths[:3]:
                print(f"    {path.path_id}: {path.error}")
        
        # Should successfully explore most budget boundary paths
        assert success_rate > 0.7, f"Budget boundary exploration success rate insufficient: {success_rate:.2%}"
        
        # Should explore diverse constraint combinations
        constraint_types = set()
        for path in paths:
            for constraint in path.constraints:
                constraint_types.add(constraint.constraint_type)
        
        assert len(constraint_types) >= 3, f"Should explore diverse constraint types, found: {constraint_types}"
    
    def test_chunking_boundary_concolic_exploration(self, concolic_engine):
        """Test concolic exploration of chunking boundaries."""
        
        # Create target function wrapper
        chunker_target = PackRepoChunkerTarget()
        
        def chunking_target_function(test_input):
            return chunker_target.execute(test_input)
        
        # Explore chunking boundary paths
        paths = concolic_engine.explore_chunking_boundary_paths(chunking_target_function)
        
        # Analyze results
        assert len(paths) > 0, "Should explore chunking boundary paths"
        
        successful_paths = [p for p in paths if p.error is None]
        failed_paths = [p for p in paths if p.error is not None]
        
        success_rate = len(successful_paths) / len(paths)
        
        print(f"Chunking boundary concolic exploration results:")
        print(f"  Total paths explored: {len(paths)}")
        print(f"  Successful paths: {len(successful_paths)}")
        print(f"  Failed paths: {len(failed_paths)}")
        print(f"  Success rate: {success_rate:.2%}")
        
        # Analyze chunking results
        if successful_paths:
            print(f"  Sample chunking results:")
            for path in successful_paths[:3]:
                result = path.execution_result
                num_chunks = len(result) if isinstance(result, list) else 0
                print(f"    {path.path_id}: {num_chunks} chunks produced")
        
        # Should handle most chunking boundary cases
        assert success_rate > 0.8, f"Chunking boundary exploration success rate insufficient: {success_rate:.2%}"
        
        # Validate chunking logic
        for path in successful_paths:
            if isinstance(path.execution_result, list):
                chunks = path.execution_result
                # Each chunk should have valid line ranges
                for chunk in chunks:
                    if hasattr(chunk, 'start_line') and hasattr(chunk, 'end_line'):
                        assert chunk.start_line >= 1, f"Invalid start line: {chunk.start_line}"
                        assert chunk.end_line >= chunk.start_line, f"Invalid line range: {chunk.start_line}-{chunk.end_line}"
    
    def test_selection_boundary_concolic_exploration(self, concolic_engine):
        """Test concolic exploration of selection algorithm boundaries."""
        
        # Create target function wrapper
        selector_target = PackRepoSelectorTarget()
        
        def selection_target_function(test_input):
            return selector_target.execute(test_input)
        
        # Explore selection boundary paths
        paths = concolic_engine.explore_selection_boundary_paths(selection_target_function)
        
        # Analyze results
        assert len(paths) > 0, "Should explore selection boundary paths"
        
        successful_paths = [p for p in paths if p.error is None]
        failed_paths = [p for p in paths if p.error is not None]
        
        success_rate = len(successful_paths) / len(paths)
        
        print(f"Selection boundary concolic exploration results:")
        print(f"  Total paths explored: {len(paths)}")
        print(f"  Successful paths: {len(successful_paths)}")
        print(f"  Failed paths: {len(failed_paths)}")
        print(f"  Success rate: {success_rate:.2%}")
        
        # Analyze selection results
        if successful_paths:
            print(f"  Sample selection results:")
            for path in successful_paths[:3]:
                result = path.execution_result
                selected_count = len(result) if isinstance(result, (list, set)) else 0
                
                # Calculate total cost of selection
                total_cost = 0
                if isinstance(result, (list, set)):
                    for item in result:
                        if isinstance(item, dict) and 'cost' in item:
                            total_cost += item['cost']
                        elif hasattr(item, 'cost'):
                            total_cost += item.cost
                
                budget = path.concrete_inputs.get('budget', 0)
                print(f"    {path.path_id}: {selected_count} items, cost={total_cost}, budget={budget}")
        
        # Should handle most selection boundary cases
        assert success_rate > 0.7, f"Selection boundary exploration success rate insufficient: {success_rate:.2%}"
        
        # Validate selection logic - selected items should not exceed budget
        for path in successful_paths:
            if isinstance(path.execution_result, (list, set)) and 'budget' in path.concrete_inputs:
                budget = path.concrete_inputs['budget']
                total_cost = 0
                
                for item in path.execution_result:
                    if isinstance(item, dict) and 'cost' in item:
                        total_cost += item['cost']
                    elif hasattr(item, 'cost'):
                        total_cost += item.cost
                
                # Allow small tolerance for floating point arithmetic
                assert total_cost <= budget * 1.01, f"Budget violation: {total_cost} > {budget} in path {path.path_id}"
    
    def test_comprehensive_boundary_coverage(self, concolic_engine):
        """Test comprehensive boundary condition coverage."""
        
        selector_target = PackRepoSelectorTarget()
        chunker_target = PackRepoChunkerTarget()
        
        # Explore all boundary types
        budget_paths = concolic_engine.explore_budget_boundary_paths(
            lambda x: selector_target.execute(x)
        )
        
        chunking_paths = concolic_engine.explore_chunking_boundary_paths(
            lambda x: chunker_target.execute(x)
        )
        
        selection_paths = concolic_engine.explore_selection_boundary_paths(
            lambda x: selector_target.execute(x)
        )
        
        # Generate comprehensive coverage report
        report = concolic_engine.generate_path_coverage_report()
        
        print(f"Comprehensive boundary coverage report:")
        print(f"  Total paths explored: {report['total_paths_explored']}")
        print(f"  Success rate: {report['success_rate']:.2%}")
        print(f"  Constraint types covered: {report['constraint_types_covered']}")
        print(f"  Variables analyzed: {report['variables_analyzed']}")
        print(f"  Path diversity score: {report['path_diversity_score']:.2f}")
        
        # Validate comprehensive coverage
        assert report['total_paths_explored'] >= 15, f"Insufficient path exploration: {report['total_paths_explored']}"
        assert report['success_rate'] > 0.7, f"Overall success rate too low: {report['success_rate']:.2%}"
        assert len(report['constraint_types_covered']) >= 3, f"Insufficient constraint type coverage: {report['constraint_types_covered']}"
        assert report['path_diversity_score'] > 0.8, f"Insufficient path diversity: {report['path_diversity_score']:.2f}"
        
        # Show coverage statistics
        total_boundary_paths = len(budget_paths) + len(chunking_paths) + len(selection_paths)
        successful_boundary_paths = sum(1 for paths in [budget_paths, chunking_paths, selection_paths]
                                      for path in paths if path.error is None)
        
        boundary_success_rate = successful_boundary_paths / total_boundary_paths
        
        print(f"  Boundary-specific results:")
        print(f"    Budget paths: {len(budget_paths)}")
        print(f"    Chunking paths: {len(chunking_paths)}")
        print(f"    Selection paths: {len(selection_paths)}")
        print(f"    Overall boundary success rate: {boundary_success_rate:.2%}")
        
        assert boundary_success_rate > 0.75, f"Boundary-specific success rate insufficient: {boundary_success_rate:.2%}"
    
    @pytest.mark.slow
    def test_concolic_fuzzing_integration(self, concolic_engine):
        """Integration test combining concolic exploration with fuzzing."""
        
        from .fuzzer_engine import FuzzerEngine
        
        # Create fuzzing engine
        fuzzer = FuzzerEngine(seed=98765)
        
        # Use concolic exploration results to guide fuzzing
        selector_target = PackRepoSelectorTarget()
        
        # First, explore paths with concolic execution
        budget_paths = concolic_engine.explore_budget_boundary_paths(
            lambda x: selector_target.execute(x)
        )
        
        # Extract interesting boundary values from concolic exploration
        boundary_budgets = set()
        boundary_costs = set()
        boundary_scores = set()
        
        for path in budget_paths:
            if 'budget' in path.concrete_inputs:
                boundary_budgets.add(path.concrete_inputs['budget'])
            
            if 'chunks' in path.concrete_inputs:
                for chunk in path.concrete_inputs['chunks']:
                    if 'cost' in chunk:
                        boundary_costs.add(chunk['cost'])
                    if 'score' in chunk:
                        boundary_scores.add(chunk['score'])
        
        # Use boundary values to guide fuzzing
        fuzzing_test_cases = []
        
        for budget in list(boundary_budgets)[:5]:  # Limit for performance
            for cost in list(boundary_costs)[:3]:
                for score in list(boundary_scores)[:3]:
                    test_case = {
                        'chunks': [{'id': 'guided_chunk', 'cost': cost, 'score': score}],
                        'budget': budget
                    }
                    fuzzing_test_cases.append(test_case)
        
        # Execute guided fuzzing test cases
        guided_results = []
        for test_case in fuzzing_test_cases:
            try:
                result = selector_target.execute(test_case)
                guided_results.append({
                    'test_case': test_case,
                    'result': result,
                    'success': True
                })
            except Exception as e:
                guided_results.append({
                    'test_case': test_case,
                    'error': str(e),
                    'success': False
                })
        
        # Analyze guided fuzzing results
        guided_success_count = sum(1 for r in guided_results if r['success'])
        guided_success_rate = guided_success_count / len(guided_results)
        
        print(f"Concolic-guided fuzzing integration results:")
        print(f"  Boundary values extracted: {len(boundary_budgets)} budgets, {len(boundary_costs)} costs, {len(boundary_scores)} scores")
        print(f"  Guided test cases: {len(fuzzing_test_cases)}")
        print(f"  Guided fuzzing success rate: {guided_success_rate:.2%}")
        
        # Concolic-guided fuzzing should have high success rate
        assert guided_success_rate > 0.8, f"Guided fuzzing success rate insufficient: {guided_success_rate:.2%}"
        
        # Compare with random fuzzing
        random_test_cases = []
        for _ in range(len(fuzzing_test_cases)):
            random_test_case = {
                'chunks': [{
                    'id': 'random_chunk',
                    'cost': random.randint(1, 200),
                    'score': random.uniform(0, 1)
                }],
                'budget': random.randint(50, 300)
            }
            random_test_cases.append(random_test_case)
        
        random_results = []
        for test_case in random_test_cases:
            try:
                result = selector_target.execute(test_case)
                random_results.append(True)
            except Exception:
                random_results.append(False)
        
        random_success_rate = sum(random_results) / len(random_results)
        
        print(f"  Random fuzzing success rate: {random_success_rate:.2%}")
        print(f"  Improvement from concolic guidance: {guided_success_rate - random_success_rate:.2%}")
        
        # Guided fuzzing should perform better than random
        assert guided_success_rate >= random_success_rate, "Concolic-guided fuzzing should not perform worse than random"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
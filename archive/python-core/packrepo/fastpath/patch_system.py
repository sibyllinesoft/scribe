"""
Two-Pass Speculate→Patch System (Workstream D)

Implements speculative file selection with intelligent patching:
- Pass 1: Fast speculative selection based on heuristics and rules
- Pass 2: Patch gaps in selection using rules-only criteria
- Rules-based gap detection ensures coverage of critical functionality
- Flag-guarded integration with budget-aware execution

Research-grade implementation for publication standards.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from .fast_scan import ScanResult
from .feature_flags import get_feature_flags
from ..packer.tokenizer import estimate_tokens_scan_result


class InclusionRule(Enum):
    """Rules for determining file inclusion necessity."""
    CRITICAL_DEPENDENCY = "critical_dependency"    # Required by selected files
    ENTRY_POINT = "entry_point"                   # Application entry points
    CONFIG_REQUIRED = "config_required"           # Configuration files
    TEST_COVERAGE = "test_coverage"               # Test files for selected code
    API_COMPLETENESS = "api_completeness"         # Complete API surface
    BUILD_SYSTEM = "build_system"                 # Build and deployment files
    DOCUMENTATION_CORE = "documentation_core"     # Core documentation
    TYPE_DEFINITIONS = "type_definitions"         # Type definitions and schemas


@dataclass
class SpeculationResult:
    """Result of first-pass speculative selection."""
    selected_files: List[ScanResult]
    selection_scores: Dict[str, float]
    budget_used: int
    budget_remaining: int
    coverage_gaps: List[str]  # Identified gaps in coverage


@dataclass
class PatchCandidate:
    """Candidate file for patching gaps in selection."""
    scan_result: ScanResult
    inclusion_rules: List[InclusionRule]
    patch_priority: float
    estimated_value: float
    token_cost: int
    justification: str


@dataclass
class PatchResult:
    """Result of second-pass patching."""
    patched_files: List[ScanResult]
    patch_rules_applied: List[InclusionRule]
    additional_budget_used: int
    coverage_improvement: float
    final_file_count: int


class RulesEngine:
    """
    Rules-based analysis engine for gap detection and patch selection.
    
    Implements sophisticated rules for determining what files are necessary
    for complete functionality coverage, regardless of heuristic scores.
    """
    
    def __init__(self):
        # Cache for expensive computations
        self._dependency_cache: Dict[str, Set[str]] = {}
        self._api_surface_cache: Dict[str, Set[str]] = {}
        
    def analyze_coverage_gaps(
        self, 
        selected_files: List[ScanResult], 
        all_files: List[ScanResult]
    ) -> List[str]:
        """
        Analyze what critical functionality gaps exist in the selection.
        
        Returns list of gap descriptions that need to be addressed.
        """
        selected_paths = {f.stats.path for f in selected_files}
        gaps = []
        
        # Gap 1: Missing critical dependencies
        missing_deps = self._find_missing_dependencies(selected_files, all_files)
        if missing_deps:
            gaps.append(f"Missing {len(missing_deps)} critical dependencies")
            
        # Gap 2: Incomplete API surface
        incomplete_apis = self._find_incomplete_api_surface(selected_files, all_files)
        if incomplete_apis:
            gaps.append(f"Incomplete API surface: {len(incomplete_apis)} modules")
            
        # Gap 3: Missing entry points
        missing_entry_points = self._find_missing_entry_points(selected_files, all_files)
        if missing_entry_points:
            gaps.append(f"Missing {len(missing_entry_points)} entry points")
            
        # Gap 4: Configuration gaps
        config_gaps = self._find_configuration_gaps(selected_files, all_files)
        if config_gaps:
            gaps.append(f"Missing {len(config_gaps)} config files")
            
        # Gap 5: Build system gaps
        build_gaps = self._find_build_system_gaps(selected_files, all_files)
        if build_gaps:
            gaps.append(f"Missing {len(build_gaps)} build files")
            
        # Gap 6: Test coverage gaps
        test_gaps = self._find_test_coverage_gaps(selected_files, all_files)
        if test_gaps:
            gaps.append(f"Missing tests for {len(test_gaps)} modules")
            
        return gaps
        
    def _find_missing_dependencies(self, selected_files: List[ScanResult], all_files: List[ScanResult]) -> Set[str]:
        """Find files that are dependencies of selected files but not included."""
        selected_paths = {f.stats.path for f in selected_files}
        missing_deps = set()
        
        for selected_file in selected_files:
            deps = self._get_file_dependencies(selected_file, all_files)
            for dep in deps:
                if dep not in selected_paths:
                    missing_deps.add(dep)
                    
        return missing_deps
        
    def _get_file_dependencies(self, scan_result: ScanResult, all_files: List[ScanResult]) -> Set[str]:
        """Get dependencies for a file (cached)."""
        file_path = scan_result.stats.path
        
        if file_path in self._dependency_cache:
            return self._dependency_cache[file_path]
            
        dependencies = set()
        
        if scan_result.imports and scan_result.imports.imports:
            # Map import strings to file paths
            file_map = {f.stats.path: f for f in all_files}
            
            for import_str in scan_result.imports.imports:
                resolved_path = self._resolve_import_to_file(import_str, file_path, file_map)
                if resolved_path:
                    dependencies.add(resolved_path)
                    
        self._dependency_cache[file_path] = dependencies
        return dependencies
        
    def _resolve_import_to_file(self, import_str: str, current_file: str, file_map: Dict[str, ScanResult]) -> Optional[str]:
        """Resolve import string to actual file path (simplified)."""
        # This is a simplified version - production would be more sophisticated
        import_lower = import_str.lower()
        
        # Direct path matching
        for file_path in file_map:
            if import_lower in file_path.lower():
                return file_path
                
        return None
        
    def _find_incomplete_api_surface(self, selected_files: List[ScanResult], all_files: List[ScanResult]) -> Set[str]:
        """Find modules where only part of the public API is included."""
        selected_paths = {f.stats.path for f in selected_files}
        incomplete_modules = set()
        
        # Group files by module/package
        modules = defaultdict(list)
        for file_result in all_files:
            # Extract module name from path
            module_name = self._extract_module_name(file_result.stats.path)
            modules[module_name].append(file_result)
            
        # Check each module for completeness
        for module_name, module_files in modules.items():
            if len(module_files) <= 1:
                continue  # Single file modules are complete if included
                
            selected_in_module = [f for f in module_files if f.stats.path in selected_paths]
            
            # If some but not all files from module are selected
            if 0 < len(selected_in_module) < len(module_files):
                # Check if missing files contain important exports
                missing_files = [f for f in module_files if f.stats.path not in selected_paths]
                if any(self._has_public_exports(f) for f in missing_files):
                    incomplete_modules.add(module_name)
                    
        return incomplete_modules
        
    def _extract_module_name(self, file_path: str) -> str:
        """Extract module/package name from file path."""
        # Simplified module extraction
        parts = file_path.split('/')
        if len(parts) >= 2:
            return parts[-2]  # Parent directory as module name
        return "root"
        
    def _has_public_exports(self, scan_result: ScanResult) -> bool:
        """Check if file has public exports/API surface."""
        # Simplified check - would be more sophisticated in production
        if scan_result.stats.is_entrypoint:
            return True
            
        # Check for common public indicators in filename
        filename = scan_result.stats.path.split('/')[-1].lower()
        public_indicators = ['index', 'main', 'api', 'public', '__init__']
        
        return any(indicator in filename for indicator in public_indicators)
        
    def _find_missing_entry_points(self, selected_files: List[ScanResult], all_files: List[ScanResult]) -> Set[str]:
        """Find entry points that are missing from selection."""
        selected_paths = {f.stats.path for f in selected_files}
        missing_entry_points = set()
        
        for file_result in all_files:
            if file_result.stats.is_entrypoint and file_result.stats.path not in selected_paths:
                missing_entry_points.add(file_result.stats.path)
                
        return missing_entry_points
        
    def _find_configuration_gaps(self, selected_files: List[ScanResult], all_files: List[ScanResult]) -> Set[str]:
        """Find configuration files missing from selection."""
        selected_paths = {f.stats.path for f in selected_files}
        config_gaps = set()
        
        # Critical config file patterns
        critical_config_patterns = [
            'package.json', 'requirements.txt', 'cargo.toml', 'go.mod',
            'dockerfile', 'docker-compose', 'makefile',
            '.env', 'config.json', 'config.yaml'
        ]
        
        for file_result in all_files:
            filename = file_result.stats.path.split('/')[-1].lower()
            
            if (any(pattern in filename for pattern in critical_config_patterns) or
                file_result.stats.is_readme or
                file_result.stats.is_docs):
                
                if file_result.stats.path not in selected_paths:
                    config_gaps.add(file_result.stats.path)
                    
        return config_gaps
        
    def _find_build_system_gaps(self, selected_files: List[ScanResult], all_files: List[ScanResult]) -> Set[str]:
        """Find build system files missing from selection."""
        selected_paths = {f.stats.path for f in selected_files}
        build_gaps = set()
        
        build_patterns = [
            'makefile', 'dockerfile', 'docker-compose',
            '.github', '.gitlab-ci', 'jenkins', 'travis',
            'setup.py', 'setup.cfg', 'pyproject.toml',
            'package.json', 'webpack', 'rollup', 'vite'
        ]
        
        for file_result in all_files:
            path_lower = file_result.stats.path.lower()
            
            if any(pattern in path_lower for pattern in build_patterns):
                if file_result.stats.path not in selected_paths:
                    build_gaps.add(file_result.stats.path)
                    
        return build_gaps
        
    def _find_test_coverage_gaps(self, selected_files: List[ScanResult], all_files: List[ScanResult]) -> Set[str]:
        """Find test files that should be included for selected code."""
        selected_paths = {f.stats.path for f in selected_files}
        test_gaps = set()
        
        # Find code files in selection
        code_files = [f for f in selected_files if not f.stats.is_test]
        
        # For each code file, look for corresponding tests
        for code_file in code_files:
            corresponding_tests = self._find_corresponding_tests(code_file, all_files)
            for test_file in corresponding_tests:
                if test_file not in selected_paths:
                    test_gaps.add(test_file)
                    
        return test_gaps
        
    def _find_corresponding_tests(self, code_file: ScanResult, all_files: List[ScanResult]) -> List[str]:
        """Find test files corresponding to a code file."""
        corresponding_tests = []
        code_path = code_file.stats.path
        code_basename = code_path.split('/')[-1].split('.')[0]
        
        for file_result in all_files:
            if not file_result.stats.is_test:
                continue
                
            test_path = file_result.stats.path
            test_basename = test_path.split('/')[-1]
            
            # Check for common test naming patterns
            if (code_basename in test_basename or
                test_basename.replace('test_', '').replace('_test', '') == code_basename):
                corresponding_tests.append(test_path)
                
        return corresponding_tests
        
    def generate_patch_candidates(
        self, 
        selected_files: List[ScanResult], 
        all_files: List[ScanResult],
        remaining_budget: int
    ) -> List[PatchCandidate]:
        """Generate candidate files for patching coverage gaps."""
        selected_paths = {f.stats.path for f in selected_files}
        candidates = []
        
        for file_result in all_files:
            if file_result.stats.path in selected_paths:
                continue  # Already selected
                
            # Analyze what rules this file would satisfy
            applicable_rules = self._analyze_file_rules(file_result, selected_files, all_files)
            
            if not applicable_rules:
                continue  # No rules apply
                
            # Calculate patch priority and value
            patch_priority = self._calculate_patch_priority(file_result, applicable_rules)
            estimated_value = self._estimate_patch_value(file_result, applicable_rules, selected_files)
            token_cost = estimate_tokens_scan_result(file_result)
            
            # Only consider if within budget
            if token_cost <= remaining_budget:
                justification = self._generate_patch_justification(applicable_rules)
                
                candidate = PatchCandidate(
                    scan_result=file_result,
                    inclusion_rules=applicable_rules,
                    patch_priority=patch_priority,
                    estimated_value=estimated_value,
                    token_cost=token_cost,
                    justification=justification
                )
                candidates.append(candidate)
                
        # Sort by patch priority (descending)
        candidates.sort(key=lambda c: c.patch_priority, reverse=True)
        
        return candidates
        
    def _analyze_file_rules(
        self, 
        file_result: ScanResult, 
        selected_files: List[ScanResult], 
        all_files: List[ScanResult]
    ) -> List[InclusionRule]:
        """Analyze what inclusion rules apply to a file."""
        rules = []
        
        # Entry point rule
        if file_result.stats.is_entrypoint:
            rules.append(InclusionRule.ENTRY_POINT)
            
        # Critical dependency rule
        if self._is_critical_dependency(file_result, selected_files, all_files):
            rules.append(InclusionRule.CRITICAL_DEPENDENCY)
            
        # Config required rule  
        if self._is_required_config(file_result):
            rules.append(InclusionRule.CONFIG_REQUIRED)
            
        # Test coverage rule
        if file_result.stats.is_test and self._provides_test_coverage(file_result, selected_files):
            rules.append(InclusionRule.TEST_COVERAGE)
            
        # API completeness rule
        if self._completes_api_surface(file_result, selected_files, all_files):
            rules.append(InclusionRule.API_COMPLETENESS)
            
        # Build system rule
        if self._is_build_system_file(file_result):
            rules.append(InclusionRule.BUILD_SYSTEM)
            
        # Documentation core rule
        if self._is_core_documentation(file_result):
            rules.append(InclusionRule.DOCUMENTATION_CORE)
            
        # Type definitions rule
        if self._provides_type_definitions(file_result):
            rules.append(InclusionRule.TYPE_DEFINITIONS)
            
        return rules
        
    def _is_critical_dependency(self, file_result: ScanResult, selected_files: List[ScanResult], all_files: List[ScanResult]) -> bool:
        """Check if file is a critical dependency of selected files."""
        file_path = file_result.stats.path
        
        for selected_file in selected_files:
            deps = self._get_file_dependencies(selected_file, all_files)
            if file_path in deps:
                return True
                
        return False
        
    def _is_required_config(self, file_result: ScanResult) -> bool:
        """Check if file is a required configuration file."""
        filename = file_result.stats.path.split('/')[-1].lower()
        
        required_configs = [
            'package.json', 'requirements.txt', 'cargo.toml', 'go.mod',
            'pyproject.toml', 'setup.py', 'pom.xml'
        ]
        
        return any(config in filename for config in required_configs)
        
    def _provides_test_coverage(self, file_result: ScanResult, selected_files: List[ScanResult]) -> bool:
        """Check if test file provides coverage for selected code."""
        if not file_result.stats.is_test:
            return False
            
        test_basename = file_result.stats.path.split('/')[-1]
        test_name = test_basename.replace('test_', '').replace('_test', '').split('.')[0]
        
        for selected_file in selected_files:
            if selected_file.stats.is_test:
                continue
                
            code_basename = selected_file.stats.path.split('/')[-1].split('.')[0]
            if test_name == code_basename:
                return True
                
        return False
        
    def _completes_api_surface(self, file_result: ScanResult, selected_files: List[ScanResult], all_files: List[ScanResult]) -> bool:
        """Check if file completes an incomplete API surface."""
        # Simplified check - would be more sophisticated in production
        module_name = self._extract_module_name(file_result.stats.path)
        
        # Check if other files from same module are selected
        selected_paths = {f.stats.path for f in selected_files}
        module_files = [f for f in all_files if self._extract_module_name(f.stats.path) == module_name]
        
        selected_in_module = [f for f in module_files if f.stats.path in selected_paths]
        
        # If some files from module are selected and this has public exports
        return len(selected_in_module) > 0 and self._has_public_exports(file_result)
        
    def _is_build_system_file(self, file_result: ScanResult) -> bool:
        """Check if file is part of build system."""
        path_lower = file_result.stats.path.lower()
        
        build_indicators = [
            'makefile', 'dockerfile', 'docker-compose',
            '.github', '.gitlab-ci', 'jenkins',
            'webpack', 'rollup', 'vite.config'
        ]
        
        return any(indicator in path_lower for indicator in build_indicators)
        
    def _is_core_documentation(self, file_result: ScanResult) -> bool:
        """Check if file is core documentation."""
        return (file_result.stats.is_readme or 
                file_result.stats.is_docs or
                'architecture' in file_result.stats.path.lower() or
                'design' in file_result.stats.path.lower())
        
    def _provides_type_definitions(self, file_result: ScanResult) -> bool:
        """Check if file provides type definitions."""
        filename = file_result.stats.path.split('/')[-1].lower()
        
        return (filename.endswith(('.d.ts', '.types.ts')) or
                'types' in filename or
                'schema' in filename)
        
    def _calculate_patch_priority(self, file_result: ScanResult, rules: List[InclusionRule]) -> float:
        """Calculate priority score for patch candidate."""
        # Base priority by rule importance
        rule_priorities = {
            InclusionRule.CRITICAL_DEPENDENCY: 1.0,
            InclusionRule.ENTRY_POINT: 0.9,
            InclusionRule.CONFIG_REQUIRED: 0.8,
            InclusionRule.API_COMPLETENESS: 0.7,
            InclusionRule.BUILD_SYSTEM: 0.6,
            InclusionRule.TYPE_DEFINITIONS: 0.5,
            InclusionRule.TEST_COVERAGE: 0.4,
            InclusionRule.DOCUMENTATION_CORE: 0.3
        }
        
        # Sum priorities for all applicable rules
        total_priority = sum(rule_priorities.get(rule, 0.0) for rule in rules)
        
        # Boost for multiple rules
        if len(rules) > 1:
            total_priority *= (1.0 + 0.1 * (len(rules) - 1))
            
        return total_priority
        
    def _estimate_patch_value(self, file_result: ScanResult, rules: List[InclusionRule], selected_files: List[ScanResult]) -> float:
        """Estimate value added by including this file."""
        # Value based on rules and file characteristics
        base_value = len(rules) * 0.2  # Each rule adds value
        
        # Boost for small files (high value per token)
        size_factor = min(1.0, 1000.0 / max(file_result.stats.size_bytes, 100))
        
        # Boost for important file types
        if file_result.stats.is_entrypoint:
            base_value *= 1.5
        elif file_result.stats.is_readme:
            base_value *= 1.3
        elif file_result.stats.is_docs:
            base_value *= 1.1
            
        return base_value * size_factor
        
    def _generate_patch_justification(self, rules: List[InclusionRule]) -> str:
        """Generate human-readable justification for patch."""
        rule_descriptions = {
            InclusionRule.CRITICAL_DEPENDENCY: "Required dependency",
            InclusionRule.ENTRY_POINT: "Application entry point",
            InclusionRule.CONFIG_REQUIRED: "Essential configuration",
            InclusionRule.API_COMPLETENESS: "Completes API surface",
            InclusionRule.BUILD_SYSTEM: "Build/deployment file",
            InclusionRule.TYPE_DEFINITIONS: "Type definitions",
            InclusionRule.TEST_COVERAGE: "Test coverage",
            InclusionRule.DOCUMENTATION_CORE: "Core documentation"
        }
        
        descriptions = [rule_descriptions.get(rule, str(rule)) for rule in rules]
        return " + ".join(descriptions)


class TwoPassSelector:
    """
    Main orchestrator for two-pass speculate→patch selection.
    
    Coordinates:
    1. Fast speculative selection based on heuristic scores
    2. Gap analysis using rules engine
    3. Intelligent patching to fill critical gaps
    4. Budget management across both passes
    """
    
    def __init__(self):
        self.rules_engine = RulesEngine()
        
    def execute_two_pass_selection(
        self,
        scan_results: List[ScanResult],
        heuristic_scores: Dict[str, float], 
        total_budget: int,
        speculation_budget_ratio: float = 0.75
    ) -> Tuple[List[ScanResult], SpeculationResult, PatchResult]:
        """
        Execute complete two-pass selection process.
        
        Args:
            scan_results: All available files
            heuristic_scores: Heuristic importance scores for files
            total_budget: Total token budget available
            speculation_budget_ratio: Fraction of budget for speculation pass (0.75 = 75%)
            
        Returns:
            Tuple of (final_selected_files, speculation_result, patch_result)
        """
        flags = get_feature_flags()
        
        # If patch system disabled, fall back to simple selection
        if not flags.patch_enabled:
            return self._fallback_selection(scan_results, heuristic_scores, total_budget)
            
        # Phase 1: Speculative selection
        speculation_budget = int(total_budget * speculation_budget_ratio)
        speculation_result = self._execute_speculation_pass(
            scan_results, heuristic_scores, speculation_budget
        )
        
        # Phase 2: Gap analysis and patching
        remaining_budget = total_budget - speculation_result.budget_used
        patch_result = self._execute_patch_pass(
            speculation_result.selected_files,
            scan_results,
            remaining_budget
        )
        
        # Combine results
        final_selected_files = speculation_result.selected_files + patch_result.patched_files
        
        return final_selected_files, speculation_result, patch_result
        
    def _fallback_selection(
        self, 
        scan_results: List[ScanResult], 
        heuristic_scores: Dict[str, float], 
        total_budget: int
    ) -> Tuple[List[ScanResult], SpeculationResult, PatchResult]:
        """Fallback selection when patch system is disabled."""
        # Simple greedy selection by heuristic score
        scored_files = [(result, heuristic_scores.get(result.stats.path, 0.0)) 
                       for result in scan_results]
        scored_files.sort(key=lambda x: x[1], reverse=True)
        
        selected_files = []
        used_budget = 0
        
        for file_result, score in scored_files:
            estimated_tokens = estimate_tokens_scan_result(file_result)
            if used_budget + estimated_tokens <= total_budget:
                selected_files.append(file_result)
                used_budget += estimated_tokens
                
        # Create mock results for consistency
        speculation_result = SpeculationResult(
            selected_files=selected_files,
            selection_scores={f.stats.path: heuristic_scores.get(f.stats.path, 0.0) 
                            for f in selected_files},
            budget_used=used_budget,
            budget_remaining=total_budget - used_budget,
            coverage_gaps=[]
        )
        
        patch_result = PatchResult(
            patched_files=[],
            patch_rules_applied=[],
            additional_budget_used=0,
            coverage_improvement=0.0,
            final_file_count=len(selected_files)
        )
        
        return selected_files, speculation_result, patch_result
        
    def _execute_speculation_pass(
        self, 
        scan_results: List[ScanResult], 
        heuristic_scores: Dict[str, float],
        speculation_budget: int
    ) -> SpeculationResult:
        """Execute first pass: speculative selection based on heuristic scores."""
        
        # Sort files by heuristic score (descending)
        scored_files = [(result, heuristic_scores.get(result.stats.path, 0.0)) 
                       for result in scan_results]
        scored_files.sort(key=lambda x: x[1], reverse=True)
        
        # Greedy selection within speculation budget
        selected_files = []
        selection_scores = {}
        used_budget = 0
        
        for file_result, score in scored_files:
            estimated_tokens = estimate_tokens_scan_result(file_result)
            
            if used_budget + estimated_tokens <= speculation_budget:
                selected_files.append(file_result)
                selection_scores[file_result.stats.path] = score
                used_budget += estimated_tokens
                
        # Analyze coverage gaps
        coverage_gaps = self.rules_engine.analyze_coverage_gaps(selected_files, scan_results)
        
        return SpeculationResult(
            selected_files=selected_files,
            selection_scores=selection_scores,
            budget_used=used_budget,
            budget_remaining=speculation_budget - used_budget,
            coverage_gaps=coverage_gaps
        )
        
    def _execute_patch_pass(
        self,
        selected_files: List[ScanResult],
        all_files: List[ScanResult], 
        remaining_budget: int
    ) -> PatchResult:
        """Execute second pass: patch gaps using rules-only criteria."""
        
        if remaining_budget <= 0:
            return PatchResult(
                patched_files=[],
                patch_rules_applied=[],
                additional_budget_used=0,
                coverage_improvement=0.0,
                final_file_count=len(selected_files)
            )
            
        # Generate patch candidates
        patch_candidates = self.rules_engine.generate_patch_candidates(
            selected_files, all_files, remaining_budget
        )
        
        # Greedy selection of patch candidates
        patched_files = []
        rules_applied = set()
        budget_used = 0
        
        for candidate in patch_candidates:
            if budget_used + candidate.token_cost <= remaining_budget:
                patched_files.append(candidate.scan_result)
                rules_applied.update(candidate.inclusion_rules)
                budget_used += candidate.token_cost
                
        # Calculate coverage improvement
        initial_gaps = len(self.rules_engine.analyze_coverage_gaps(selected_files, all_files))
        final_gaps = len(self.rules_engine.analyze_coverage_gaps(
            selected_files + patched_files, all_files
        ))
        
        coverage_improvement = (initial_gaps - final_gaps) / max(initial_gaps, 1)
        
        return PatchResult(
            patched_files=patched_files,
            patch_rules_applied=list(rules_applied),
            additional_budget_used=budget_used,
            coverage_improvement=coverage_improvement,
            final_file_count=len(selected_files) + len(patched_files)
        )


def create_two_pass_selector() -> TwoPassSelector:
    """Create a TwoPassSelector instance."""
    return TwoPassSelector()
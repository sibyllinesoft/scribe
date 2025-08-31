"""
Quotas + Density-Greedy Selection Algorithm (Workstream A)

Implements category-aware budget allocation with density-greedy selection to achieve:
- Config files: ≥95% recall rate
- Entry points and Examples: ≤10% budget allocation combined
- Maximum importance density within budget constraints

Research-grade implementation for publication standards.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from .fast_scan import ScanResult
from .feature_flags import get_feature_flags
from ..packer.tokenizer import estimate_tokens_scan_result


class FileCategory(Enum):
    """File category classification for quota allocation."""
    CONFIG = "config"
    ENTRY = "entry" 
    EXAMPLES = "examples"
    GENERAL = "general"


@dataclass
class CategoryQuota:
    """Budget quota configuration for a file category."""
    category: FileCategory
    min_budget_pct: float = 0.0  # Minimum budget percentage reserved
    max_budget_pct: float = 100.0  # Maximum budget percentage allowed
    recall_target: float = 0.0  # Recall target (0.0-1.0, 0 means no target)
    priority_multiplier: float = 1.0  # Priority boost for this category


@dataclass 
class QuotaAllocation:
    """Actual budget allocation result for a category."""
    category: FileCategory
    allocated_budget: int
    used_budget: int
    file_count: int
    recall_achieved: float
    density_score: float


class CategoryDetector:
    """Detects file categories for quota allocation."""
    
    # Config file patterns
    CONFIG_PATTERNS = {
        # Configuration files
        '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
        # Build and dependency files  
        'package.json', 'requirements.txt', 'pyproject.toml', 'cargo.toml',
        'setup.py', 'setup.cfg', 'makefile', 'dockerfile', 'docker-compose.yml',
        # CI/CD configuration
        '.github', '.gitlab-ci.yml', '.travis.yml', '.circleci',
        # IDE and tool configuration
        '.vscode', '.idea', '.editorconfig', 'tsconfig.json', 'tslint.json',
        'eslint.json', '.eslintrc', '.prettierrc', 'jest.config.js'
    }
    
    # Entry point patterns
    ENTRY_PATTERNS = {
        'main.py', '__main__.py', 'app.py', 'server.py', 'index.py',
        'main.js', 'index.js', 'app.js', 'server.js', 'index.ts', 'main.ts',
        'main.go', 'main.rs', 'lib.rs', 'mod.rs'
    }
    
    # Example/demo patterns
    EXAMPLES_PATTERNS = {
        'example', 'examples', 'demo', 'demos', 'sample', 'samples',
        'tutorial', 'tutorials', 'test', 'tests', 'spec', 'specs',
        'benchmark', 'benchmarks'
    }
    
    def detect_category(self, scan_result: ScanResult) -> FileCategory:
        """Detect the category of a file based on its scan result."""
        path = scan_result.stats.path.lower()
        filename = scan_result.stats.path.split('/')[-1].lower()
        
        # Check for config files
        if self._is_config_file(path, filename):
            return FileCategory.CONFIG
            
        # Check for entry points
        if self._is_entry_file(path, filename, scan_result):
            return FileCategory.ENTRY
            
        # Check for examples
        if self._is_examples_file(path, filename):
            return FileCategory.EXAMPLES
            
        return FileCategory.GENERAL
    
    def _is_config_file(self, path: str, filename: str) -> bool:
        """Check if file is a configuration file."""
        # Check file extensions and names
        for pattern in self.CONFIG_PATTERNS:
            if pattern in filename or pattern in path:
                return True
        return False
        
    def _is_entry_file(self, path: str, filename: str, scan_result: ScanResult) -> bool:
        """Check if file is an entry point."""
        # Check explicit entry point markers
        if scan_result.stats.is_entrypoint:
            return True
            
        # Check filename patterns
        for pattern in self.ENTRY_PATTERNS:
            if filename == pattern:
                return True
                
        # Check for main function or entry point indicators
        if hasattr(scan_result, 'has_main_function') and scan_result.has_main_function:
            return True
            
        return False
        
    def _is_examples_file(self, path: str, filename: str) -> bool:
        """Check if file is examples/demos/tests."""
        for pattern in self.EXAMPLES_PATTERNS:
            if pattern in path or pattern in filename:
                return True
        return False


class QuotaManager:
    """
    Manages budget quotas and density-greedy selection.
    
    Implements category-aware selection algorithm:
    1. Classify files into categories (Config, Entry, Examples, General)
    2. Allocate budget quotas based on category priorities
    3. Apply density-greedy selection within each category
    4. Optimize for recall targets and budget constraints
    """
    
    def __init__(self, total_budget: int):
        self.total_budget = total_budget
        self.detector = CategoryDetector()
        
        # Default quota configuration (research-optimized)
        self.category_quotas = {
            FileCategory.CONFIG: CategoryQuota(
                category=FileCategory.CONFIG,
                min_budget_pct=15.0,  # Reserve at least 15% for config
                max_budget_pct=30.0,  # Cap at 30% to avoid over-allocation
                recall_target=0.95,   # 95% recall target for config files
                priority_multiplier=2.0  # High priority for config files
            ),
            FileCategory.ENTRY: CategoryQuota(
                category=FileCategory.ENTRY, 
                min_budget_pct=2.0,   # Minimum for entry points
                max_budget_pct=7.0,   # Max 7% for entry points
                recall_target=0.90,   # High recall for entry points
                priority_multiplier=1.8
            ),
            FileCategory.EXAMPLES: CategoryQuota(
                category=FileCategory.EXAMPLES,
                min_budget_pct=1.0,   # Small allocation for examples
                max_budget_pct=3.0,   # Max 3% for examples  
                recall_target=0.0,    # No recall target for examples
                priority_multiplier=0.5  # Lower priority
            ),
            FileCategory.GENERAL: CategoryQuota(
                category=FileCategory.GENERAL,
                min_budget_pct=60.0,  # Most budget goes to general files
                max_budget_pct=82.0,  # Leave room for other categories
                recall_target=0.0,    # No specific recall target
                priority_multiplier=1.0
            )
        }
        
    def classify_files(self, scan_results: List[ScanResult]) -> Dict[FileCategory, List[ScanResult]]:
        """Classify files into categories."""
        categorized = defaultdict(list)
        
        for result in scan_results:
            category = self.detector.detect_category(result)
            categorized[category].append(result)
            
        return dict(categorized)
    
    def calculate_density_score(self, scan_result: ScanResult, heuristic_score: float) -> float:
        """
        Calculate density score (importance per token).
        
        Density = importance_score / token_cost
        Higher density means more importance per token spent.
        """
        # Estimate token cost using centralized utility
        estimated_tokens = estimate_tokens_scan_result(scan_result)
        
        # Avoid division by zero
        if estimated_tokens <= 0:
            estimated_tokens = 1
            
        density = heuristic_score / estimated_tokens
        
        # Apply category priority multiplier
        category = self.detector.detect_category(scan_result)
        quota = self.category_quotas.get(category)
        if quota:
            density *= quota.priority_multiplier
            
        return density
    
    def select_files_density_greedy(
        self, 
        categorized_files: Dict[FileCategory, List[ScanResult]],
        heuristic_scores: Dict[str, float]
    ) -> Tuple[List[ScanResult], Dict[FileCategory, QuotaAllocation]]:
        """
        Apply density-greedy selection algorithm with quotas.
        
        Algorithm:
        1. Calculate budget allocation per category
        2. Compute density scores for all files in each category
        3. Greedily select highest density files within budget constraints
        4. Ensure recall targets are met for critical categories
        """
        selected_files = []
        allocations = {}
        remaining_budget = self.total_budget
        
        # Phase 1: Allocate minimum budgets
        min_allocations = {}
        for category, quota in self.category_quotas.items():
            if category not in categorized_files:
                continue
                
            min_budget = int(self.total_budget * quota.min_budget_pct / 100.0)
            min_allocations[category] = min_budget
            remaining_budget -= min_budget
        
        # Phase 2: Distribute remaining budget based on demand and priority
        additional_allocations = self._distribute_remaining_budget(
            categorized_files, heuristic_scores, remaining_budget
        )
        
        # Phase 3: Select files within each category using density-greedy
        for category, files in categorized_files.items():
            if category not in self.category_quotas:
                continue
                
            quota = self.category_quotas[category]
            allocated_budget = min_allocations.get(category, 0) + additional_allocations.get(category, 0)
            
            # Select files for this category
            selected, allocation = self._select_category_files(
                category, files, allocated_budget, quota, heuristic_scores
            )
            
            selected_files.extend(selected)
            allocations[category] = allocation
            
        return selected_files, allocations
    
    def _distribute_remaining_budget(
        self, 
        categorized_files: Dict[FileCategory, List[ScanResult]],
        heuristic_scores: Dict[str, float],
        remaining_budget: int
    ) -> Dict[FileCategory, int]:
        """Distribute remaining budget based on category demands and priorities."""
        additional_allocations = defaultdict(int)
        
        # Calculate demand scores for each category
        category_demands = {}
        for category, files in categorized_files.items():
            if category not in self.category_quotas:
                continue
                
            quota = self.category_quotas[category]
            
            # Calculate total value density for this category
            total_density = 0.0
            for file_result in files:
                heuristic_score = heuristic_scores.get(file_result.stats.path, 0.0)
                density = self.calculate_density_score(file_result, heuristic_score)
                total_density += density
                
            # Weight by priority multiplier and file count
            demand_score = total_density * quota.priority_multiplier * math.log(len(files) + 1)
            category_demands[category] = demand_score
        
        # Distribute remaining budget proportionally to demand
        total_demand = sum(category_demands.values())
        if total_demand > 0:
            for category, demand in category_demands.items():
                proportion = demand / total_demand
                additional_budget = int(remaining_budget * proportion)
                
                # Respect maximum budget constraints
                quota = self.category_quotas[category]
                max_budget = int(self.total_budget * quota.max_budget_pct / 100.0)
                min_budget = int(self.total_budget * quota.min_budget_pct / 100.0)
                
                # Don't exceed maximum allocation
                current_allocation = min_budget + additional_budget
                if current_allocation > max_budget:
                    additional_budget = max_budget - min_budget
                    
                additional_allocations[category] = max(0, additional_budget)
        
        return additional_allocations
    
    def _select_category_files(
        self,
        category: FileCategory,
        files: List[ScanResult], 
        allocated_budget: int,
        quota: CategoryQuota,
        heuristic_scores: Dict[str, float]
    ) -> Tuple[List[ScanResult], QuotaAllocation]:
        """Select files within a category using density-greedy algorithm."""
        
        # Calculate density scores for all files in category
        file_densities = []
        for file_result in files:
            heuristic_score = heuristic_scores.get(file_result.stats.path, 0.0)
            density = self.calculate_density_score(file_result, heuristic_score)
            estimated_tokens = estimate_tokens_scan_result(file_result)
            file_densities.append((file_result, density, heuristic_score, estimated_tokens))
        
        # Sort by density (descending)
        file_densities.sort(key=lambda x: x[1], reverse=True)
        
        # Greedy selection within budget
        selected = []
        used_budget = 0
        total_importance = 0.0
        
        for file_result, density, importance, tokens in file_densities:
            if used_budget + tokens <= allocated_budget:
                selected.append(file_result)
                used_budget += tokens
                total_importance += importance
            elif quota.recall_target > 0:
                # For categories with recall targets, try to fit more critical files
                # even if it means going slightly over budget
                importance_threshold = self._calculate_importance_threshold(
                    [x[2] for x in file_densities], quota.recall_target
                )
                if importance >= importance_threshold and used_budget + tokens <= allocated_budget * 1.05:
                    selected.append(file_result)
                    used_budget += tokens
                    total_importance += importance
                    
        # Calculate achieved recall
        if quota.recall_target > 0 and len(files) > 0:
            # Recall = selected high-importance files / total high-importance files
            importance_threshold = self._calculate_importance_threshold(
                [heuristic_scores.get(f.stats.path, 0.0) for f in files], 
                quota.recall_target
            )
            high_importance_files = [f for f in files 
                                   if heuristic_scores.get(f.stats.path, 0.0) >= importance_threshold]
            selected_high_importance = [f for f in selected
                                      if heuristic_scores.get(f.stats.path, 0.0) >= importance_threshold]
            
            achieved_recall = len(selected_high_importance) / max(len(high_importance_files), 1)
        else:
            achieved_recall = len(selected) / max(len(files), 1)  # Selection ratio
        
        # Calculate density score for selected set
        if used_budget > 0:
            density_score = total_importance / used_budget
        else:
            density_score = 0.0
        
        allocation = QuotaAllocation(
            category=category,
            allocated_budget=allocated_budget,
            used_budget=used_budget,
            file_count=len(selected),
            recall_achieved=achieved_recall,
            density_score=density_score
        )
        
        return selected, allocation
    
    def _calculate_importance_threshold(self, importance_scores: List[float], recall_target: float) -> float:
        """Calculate importance threshold for achieving target recall."""
        if not importance_scores:
            return 0.0
            
        # Sort scores in descending order
        sorted_scores = sorted(importance_scores, reverse=True)
        
        # Find threshold that captures top recall_target fraction
        target_count = int(len(sorted_scores) * recall_target)
        target_count = max(1, min(target_count, len(sorted_scores)))
        
        threshold_index = target_count - 1
        return sorted_scores[threshold_index]
    
    def apply_quotas_selection(
        self, 
        scan_results: List[ScanResult],
        heuristic_scores: Dict[str, float]
    ) -> Tuple[List[ScanResult], Dict[FileCategory, QuotaAllocation]]:
        """
        Main entry point for quotas-based selection.
        
        Returns:
            Tuple of (selected_files, allocation_stats)
        """
        flags = get_feature_flags()
        
        # Only apply quotas if feature is enabled
        if not flags.quotas_enabled:
            # Fallback to simple top-K selection based on heuristic scores
            scored_files = [(result, heuristic_scores.get(result.stats.path, 0.0)) 
                           for result in scan_results]
            scored_files.sort(key=lambda x: x[1], reverse=True)
            
            # Select files within budget
            selected = []
            used_budget = 0
            for result, score in scored_files:
                estimated_tokens = estimate_tokens_scan_result(result)
                if used_budget + estimated_tokens <= self.total_budget:
                    selected.append(result)
                    used_budget += estimated_tokens
                    
            return selected, {}
        
        # Apply quotas-based selection
        categorized_files = self.classify_files(scan_results)
        return self.select_files_density_greedy(categorized_files, heuristic_scores)


def create_quota_manager(total_budget: int) -> QuotaManager:
    """Create a QuotaManager instance."""
    return QuotaManager(total_budget)
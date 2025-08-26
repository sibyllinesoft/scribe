"""
Router Guard + Thompson Sampling Bandit (Workstream E)

Implements intelligent routing with adaptive selection strategy:
- Router Guard: Context-aware routing to optimal selection algorithms
- Thompson Sampling: Bayesian bandit for learning best strategies per context
- Multi-armed bandit with contextual features (repo type, size, query type)
- Performance feedback loop for continuous optimization

Research-grade implementation for publication standards.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .feature_flags import get_feature_flags


class SelectionAlgorithm(Enum):
    """Available selection algorithms for bandit routing."""
    HEURISTIC_ONLY = "heuristic_only"
    QUOTAS_DENSITY = "quotas_density"
    CENTRALITY_ENHANCED = "centrality_enhanced"  
    TWO_PASS_PATCH = "two_pass_patch"
    DEMOTION_HYBRID = "demotion_hybrid"
    COMBINED_V5 = "combined_v5"


class ContextFeature(Enum):
    """Context features for bandit learning."""
    REPO_SIZE = "repo_size"           # Small/Medium/Large
    REPO_LANGUAGE = "repo_language"   # Primary programming language
    QUERY_TYPE = "query_type"         # Comprehension/Search/Debug
    FILE_COUNT = "file_count"         # Number of files in repo
    COMPLEXITY = "complexity"         # Repository complexity score
    DOMAIN = "domain"                 # Application domain (web, ML, systems, etc.)


@dataclass
class ContextVector:
    """Context representation for bandit decision making."""
    repo_size: str = "medium"         # small/medium/large
    repo_language: str = "python"     # primary language
    query_type: str = "comprehension" # comprehension/search/debug
    file_count: int = 100
    complexity_score: float = 0.5     # 0-1 complexity score
    domain: str = "general"           # application domain
    
    def to_feature_vector(self) -> List[float]:
        """Convert context to numerical feature vector for bandit."""
        features = []
        
        # Repo size encoding (one-hot)
        size_mapping = {"small": [1,0,0], "medium": [0,1,0], "large": [0,0,1]}
        features.extend(size_mapping.get(self.repo_size, [0,1,0]))
        
        # Language encoding (simplified - could be expanded)
        lang_mapping = {
            "python": [1,0,0,0], "javascript": [0,1,0,0], 
            "typescript": [0,0,1,0], "go": [0,0,0,1]
        }
        features.extend(lang_mapping.get(self.repo_language, [1,0,0,0]))
        
        # Query type encoding
        query_mapping = {
            "comprehension": [1,0,0], "search": [0,1,0], "debug": [0,0,1]
        }
        features.extend(query_mapping.get(self.query_type, [1,0,0]))
        
        # Numerical features (normalized)
        features.append(min(self.file_count / 1000.0, 1.0))  # Normalized file count
        features.append(self.complexity_score)
        
        # Domain encoding (simplified)
        domain_mapping = {
            "web": [1,0,0], "ml": [0,1,0], "systems": [0,0,1]
        }
        features.extend(domain_mapping.get(self.domain, [0,0,0]))
        
        return features


@dataclass 
class AlgorithmArm:
    """Bandit arm representing a selection algorithm."""
    algorithm: SelectionAlgorithm
    alpha: float = 1.0    # Beta distribution parameter (successes + 1)
    beta: float = 1.0     # Beta distribution parameter (failures + 1)
    total_trials: int = 0
    total_reward: float = 0.0
    recent_rewards: List[float] = field(default_factory=list)
    context_performance: Dict[str, float] = field(default_factory=dict)
    
    def sample_probability(self) -> float:
        """Sample probability from Beta posterior (Thompson sampling)."""
        return np.random.beta(self.alpha, self.beta)
    
    def update_reward(self, reward: float, context_key: str):
        """Update arm statistics with new reward."""
        self.total_trials += 1
        self.total_reward += reward
        
        # Update Beta parameters
        if reward > 0.5:  # Consider > 0.5 as success
            self.alpha += reward
        else:
            self.beta += (1.0 - reward)
            
        # Track recent performance
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 100:  # Keep last 100 rewards
            self.recent_rewards.pop(0)
            
        # Update context-specific performance
        if context_key not in self.context_performance:
            self.context_performance[context_key] = reward
        else:
            # Exponential moving average
            alpha = 0.1
            self.context_performance[context_key] = (
                alpha * reward + (1 - alpha) * self.context_performance[context_key]
            )
    
    def get_average_reward(self) -> float:
        """Get average reward over all trials."""
        return self.total_reward / max(self.total_trials, 1)
        
    def get_recent_average(self) -> float:
        """Get average reward over recent trials."""
        if not self.recent_rewards:
            return 0.5  # Default neutral performance
        return sum(self.recent_rewards) / len(self.recent_rewards)
        
    def get_context_performance(self, context_key: str) -> float:
        """Get performance for specific context."""
        return self.context_performance.get(context_key, 0.5)


@dataclass
class BanditDecision:
    """Result of bandit algorithm selection."""
    selected_algorithm: SelectionAlgorithm
    selection_probability: float
    confidence_score: float
    context_match_score: float
    exploration_factor: float
    decision_rationale: str


class ContextAnalyzer:
    """Analyzes repository context for informed routing decisions."""
    
    def __init__(self):
        self.language_cache = {}
        self.complexity_cache = {}
        
    def analyze_repository_context(self, scan_results: List[Any], query_hint: str = "") -> ContextVector:
        """Analyze repository to extract context features."""
        
        # Determine repo size
        file_count = len(scan_results)
        if file_count < 50:
            repo_size = "small"
        elif file_count < 500:
            repo_size = "medium"
        else:
            repo_size = "large"
            
        # Determine primary language
        repo_language = self._detect_primary_language(scan_results)
        
        # Determine query type from hint
        query_type = self._classify_query_type(query_hint)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(scan_results)
        
        # Determine domain
        domain = self._detect_application_domain(scan_results)
        
        return ContextVector(
            repo_size=repo_size,
            repo_language=repo_language,
            query_type=query_type,
            file_count=file_count,
            complexity_score=complexity_score,
            domain=domain
        )
        
    def _detect_primary_language(self, scan_results: List[Any]) -> str:
        """Detect primary programming language in repository."""
        language_counts = defaultdict(int)
        
        for result in scan_results:
            path = result.stats.path if hasattr(result, 'stats') else str(result)
            ext = path.lower().split('.')[-1] if '.' in path else ''
            
            # Map extensions to languages
            ext_mapping = {
                'py': 'python',
                'js': 'javascript', 'jsx': 'javascript', 'mjs': 'javascript',
                'ts': 'typescript', 'tsx': 'typescript',
                'go': 'go',
                'rs': 'rust',
                'java': 'java', 'kotlin': 'java',
                'cpp': 'cpp', 'cc': 'cpp', 'cxx': 'cpp',
                'c': 'c', 'h': 'c'
            }
            
            language = ext_mapping.get(ext)
            if language:
                language_counts[language] += 1
                
        # Return most common language
        if language_counts:
            return max(language_counts.items(), key=lambda x: x[1])[0]
        return "unknown"
        
    def _classify_query_type(self, query_hint: str) -> str:
        """Classify query type from hint string."""
        hint_lower = query_hint.lower()
        
        # Search indicators
        search_keywords = ['find', 'search', 'locate', 'where', 'grep', 'lookup']
        if any(keyword in hint_lower for keyword in search_keywords):
            return "search"
            
        # Debug indicators  
        debug_keywords = ['bug', 'error', 'issue', 'problem', 'debug', 'fix', 'crash']
        if any(keyword in hint_lower for keyword in debug_keywords):
            return "debug"
            
        # Default to comprehension
        return "comprehension"
        
    def _calculate_complexity_score(self, scan_results: List[Any]) -> float:
        """Calculate repository complexity score (0-1)."""
        if not scan_results:
            return 0.0
            
        # Factors contributing to complexity
        total_size = sum(getattr(result.stats, 'size_bytes', 1000) 
                        for result in scan_results if hasattr(result, 'stats'))
        avg_file_size = total_size / len(scan_results)
        
        # Normalize to 0-1 range
        size_complexity = min(avg_file_size / 10000.0, 1.0)  # 10KB as reference
        
        # Directory depth complexity
        max_depth = 0
        for result in scan_results:
            if hasattr(result, 'stats'):
                depth = getattr(result.stats, 'depth', 1)
                max_depth = max(max_depth, depth)
                
        depth_complexity = min(max_depth / 10.0, 1.0)  # 10 levels as reference
        
        # Language diversity complexity
        languages = set()
        for result in scan_results:
            if hasattr(result, 'stats'):
                path = result.stats.path
                ext = path.lower().split('.')[-1] if '.' in path else ''
                if ext:
                    languages.add(ext)
                    
        diversity_complexity = min(len(languages) / 10.0, 1.0)  # 10 languages as reference
        
        # Combined complexity score
        complexity = (size_complexity * 0.4 + depth_complexity * 0.3 + diversity_complexity * 0.3)
        return complexity
        
    def _detect_application_domain(self, scan_results: List[Any]) -> str:
        """Detect application domain from file patterns."""
        domain_indicators = {
            'web': ['html', 'css', 'js', 'jsx', 'vue', 'angular', 'react'],
            'ml': ['model', 'train', 'dataset', 'numpy', 'pandas', 'torch', 'tensorflow'],
            'systems': ['kernel', 'driver', 'embedded', 'firmware', 'os', 'low-level']
        }
        
        domain_scores = defaultdict(int)
        
        for result in scan_results:
            if hasattr(result, 'stats'):
                path_lower = result.stats.path.lower()
                
                for domain, indicators in domain_indicators.items():
                    for indicator in indicators:
                        if indicator in path_lower:
                            domain_scores[domain] += 1
                            
        # Return domain with highest score
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return "general"


class ThompsonSamplingBandit:
    """
    Thompson Sampling Multi-Armed Bandit for algorithm selection.
    
    Maintains Beta distributions for each algorithm's success probability
    and uses Thompson sampling for exploration-exploitation balance.
    """
    
    def __init__(self, exploration_bonus: float = 0.1):
        self.arms: Dict[SelectionAlgorithm, AlgorithmArm] = {}
        self.exploration_bonus = exploration_bonus
        self.decision_history: List[Tuple[ContextVector, SelectionAlgorithm, float]] = []
        
        # Initialize arms for all algorithms
        for algorithm in SelectionAlgorithm:
            self.arms[algorithm] = AlgorithmArm(algorithm=algorithm)
            
    def select_algorithm(self, context: ContextVector) -> BanditDecision:
        """Select algorithm using Thompson sampling with context awareness."""
        
        context_key = self._context_to_key(context)
        algorithm_scores = {}
        
        # Sample from each arm's posterior
        for algorithm, arm in self.arms.items():
            base_probability = arm.sample_probability()
            
            # Context-specific adjustment
            context_performance = arm.get_context_performance(context_key)
            context_weight = 0.3
            adjusted_probability = (
                (1 - context_weight) * base_probability +
                context_weight * context_performance
            )
            
            # Exploration bonus for under-explored arms
            exploration_bonus = self.exploration_bonus / max(math.sqrt(arm.total_trials + 1), 1.0)
            final_score = adjusted_probability + exploration_bonus
            
            algorithm_scores[algorithm] = {
                'score': final_score,
                'base_prob': base_probability,
                'context_perf': context_performance,
                'exploration': exploration_bonus
            }
            
        # Select algorithm with highest score
        selected_algorithm = max(algorithm_scores.items(), key=lambda x: x[1]['score'])[0]
        selected_scores = algorithm_scores[selected_algorithm]
        
        # Calculate confidence and rationale
        scores_list = [scores['score'] for scores in algorithm_scores.values()]
        max_score = max(scores_list)
        second_max = sorted(scores_list, reverse=True)[1] if len(scores_list) > 1 else 0
        confidence_score = (max_score - second_max) / (max_score + 1e-10)
        
        context_match_score = selected_scores['context_perf']
        exploration_factor = selected_scores['exploration']
        
        # Generate rationale
        rationale = self._generate_decision_rationale(
            selected_algorithm, context, selected_scores
        )
        
        return BanditDecision(
            selected_algorithm=selected_algorithm,
            selection_probability=selected_scores['base_prob'],
            confidence_score=confidence_score,
            context_match_score=context_match_score,
            exploration_factor=exploration_factor,
            decision_rationale=rationale
        )
        
    def update_feedback(self, context: ContextVector, algorithm: SelectionAlgorithm, reward: float):
        """Update bandit with performance feedback."""
        context_key = self._context_to_key(context)
        
        if algorithm in self.arms:
            self.arms[algorithm].update_reward(reward, context_key)
            
        # Store in decision history
        self.decision_history.append((context, algorithm, reward))
        
        # Keep history manageable
        if len(self.decision_history) > 10000:
            self.decision_history = self.decision_history[-5000:]
            
    def get_algorithm_stats(self) -> Dict[SelectionAlgorithm, Dict[str, float]]:
        """Get performance statistics for all algorithms."""
        stats = {}
        
        for algorithm, arm in self.arms.items():
            stats[algorithm] = {
                'average_reward': arm.get_average_reward(),
                'recent_average': arm.get_recent_average(),
                'total_trials': arm.total_trials,
                'success_rate': arm.alpha / (arm.alpha + arm.beta),
                'confidence': arm.alpha + arm.beta  # Higher = more confident
            }
            
        return stats
        
    def _context_to_key(self, context: ContextVector) -> str:
        """Convert context to string key for performance tracking."""
        return f"{context.repo_size}_{context.repo_language}_{context.query_type}"
        
    def _generate_decision_rationale(
        self, 
        algorithm: SelectionAlgorithm, 
        context: ContextVector, 
        scores: Dict[str, float]
    ) -> str:
        """Generate human-readable rationale for algorithm selection."""
        
        rationale_parts = []
        
        # Algorithm-specific reasoning
        if algorithm == SelectionAlgorithm.QUOTAS_DENSITY:
            rationale_parts.append("Quota-based selection for balanced coverage")
        elif algorithm == SelectionAlgorithm.CENTRALITY_ENHANCED:
            rationale_parts.append("Centrality-enhanced for dependency analysis")
        elif algorithm == SelectionAlgorithm.TWO_PASS_PATCH:
            rationale_parts.append("Two-pass selection for comprehensive coverage")
        else:
            rationale_parts.append(f"Selected {algorithm.value}")
            
        # Context reasoning
        if context.repo_size == "large":
            rationale_parts.append("optimized for large repositories")
        elif context.complexity_score > 0.7:
            rationale_parts.append("suitable for complex codebases")
            
        if context.query_type == "search":
            rationale_parts.append("enhanced for search queries")
        elif context.query_type == "debug":
            rationale_parts.append("focused on debugging workflows")
            
        # Performance reasoning
        if scores['context_perf'] > 0.7:
            rationale_parts.append("strong historical performance in this context")
        elif scores['exploration'] > 0.05:
            rationale_parts.append("exploration of less-tried approach")
            
        return " - ".join(rationale_parts)


class RouterGuard:
    """
    Router Guard system that provides intelligent algorithm routing.
    
    Combines contextual analysis with bandit learning for optimal
    algorithm selection based on repository characteristics and task type.
    """
    
    def __init__(self):
        self.context_analyzer = ContextAnalyzer()
        self.bandit = ThompsonSamplingBandit()
        self.performance_history = []
        
        # Fallback routing rules when bandit is disabled
        self.fallback_rules = {
            ("small", "python", "comprehension"): SelectionAlgorithm.HEURISTIC_ONLY,
            ("medium", "python", "comprehension"): SelectionAlgorithm.QUOTAS_DENSITY,
            ("large", "python", "comprehension"): SelectionAlgorithm.COMBINED_V5,
            ("small", "javascript", "search"): SelectionAlgorithm.CENTRALITY_ENHANCED,
            ("medium", "javascript", "search"): SelectionAlgorithm.TWO_PASS_PATCH,
            ("large", "javascript", "search"): SelectionAlgorithm.COMBINED_V5,
        }
        
    def route_selection_algorithm(
        self, 
        scan_results: List[Any], 
        query_hint: str = ""
    ) -> BanditDecision:
        """
        Route to optimal selection algorithm based on context.
        
        Returns decision with selected algorithm and reasoning.
        """
        flags = get_feature_flags()
        
        # Analyze repository context
        context = self.context_analyzer.analyze_repository_context(scan_results, query_hint)
        
        # Use bandit if enabled
        if flags.bandit_enabled:
            return self.bandit.select_algorithm(context)
        else:
            # Use fallback rules
            return self._apply_fallback_routing(context)
            
    def update_performance_feedback(
        self, 
        context: ContextVector, 
        algorithm: SelectionAlgorithm, 
        metrics: Dict[str, float]
    ):
        """Update router with performance feedback from algorithm execution."""
        
        # Calculate composite reward score (0-1)
        reward_components = {
            'qa_improvement': metrics.get('qa_improvement_pct', 0.0) / 100.0,
            'budget_efficiency': metrics.get('budget_efficiency', 0.0),
            'coverage_completeness': metrics.get('coverage_completeness', 0.0),
            'latency_penalty': 1.0 - min(metrics.get('latency_overhead_pct', 0.0) / 50.0, 1.0)
        }
        
        # Weighted combination
        weights = {'qa_improvement': 0.4, 'budget_efficiency': 0.3, 'coverage_completeness': 0.2, 'latency_penalty': 0.1}
        reward = sum(weights[k] * reward_components[k] for k in weights.keys())
        reward = max(0.0, min(1.0, reward))  # Clamp to [0,1]
        
        # Update bandit
        flags = get_feature_flags()
        if flags.bandit_enabled:
            self.bandit.update_feedback(context, algorithm, reward)
            
        # Store in performance history
        self.performance_history.append({
            'timestamp': len(self.performance_history),
            'context': context,
            'algorithm': algorithm,
            'reward': reward,
            'metrics': metrics.copy()
        })
        
    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get analytics on routing decisions and performance."""
        
        analytics = {
            'total_decisions': len(self.performance_history),
            'algorithm_usage': defaultdict(int),
            'algorithm_performance': {},
            'context_patterns': defaultdict(list)
        }
        
        # Aggregate statistics
        for entry in self.performance_history:
            algorithm = entry['algorithm']
            reward = entry['reward']
            context = entry['context']
            
            analytics['algorithm_usage'][algorithm] += 1
            
            if algorithm not in analytics['algorithm_performance']:
                analytics['algorithm_performance'][algorithm] = []
            analytics['algorithm_performance'][algorithm].append(reward)
            
            context_key = f"{context.repo_size}_{context.repo_language}"
            analytics['context_patterns'][context_key].append((algorithm, reward))
            
        # Calculate performance averages
        for algorithm, rewards in analytics['algorithm_performance'].items():
            analytics['algorithm_performance'][algorithm] = {
                'average_reward': sum(rewards) / len(rewards),
                'total_uses': len(rewards),
                'recent_average': sum(rewards[-10:]) / min(len(rewards), 10)
            }
            
        # Get bandit statistics if available
        flags = get_feature_flags()
        if flags.bandit_enabled:
            analytics['bandit_stats'] = self.bandit.get_algorithm_stats()
            
        return analytics
        
    def _apply_fallback_routing(self, context: ContextVector) -> BanditDecision:
        """Apply rule-based fallback routing when bandit is disabled."""
        
        # Create lookup key
        lookup_key = (context.repo_size, context.repo_language, context.query_type)
        
        # Try exact match
        if lookup_key in self.fallback_rules:
            selected_algorithm = self.fallback_rules[lookup_key]
        else:
            # Try partial matches
            for (size, lang, query), algorithm in self.fallback_rules.items():
                match_score = 0
                if size == context.repo_size:
                    match_score += 1
                if lang == context.repo_language:
                    match_score += 1
                if query == context.query_type:
                    match_score += 1
                    
                if match_score >= 2:  # At least 2/3 match
                    selected_algorithm = algorithm
                    break
            else:
                # Default fallback
                if context.repo_size == "large" or context.complexity_score > 0.7:
                    selected_algorithm = SelectionAlgorithm.COMBINED_V5
                elif context.query_type == "search":
                    selected_algorithm = SelectionAlgorithm.CENTRALITY_ENHANCED
                else:
                    selected_algorithm = SelectionAlgorithm.QUOTAS_DENSITY
                    
        # Create decision with rule-based rationale
        rationale = f"Rule-based routing: {context.repo_size} {context.repo_language} {context.query_type} → {selected_algorithm.value}"
        
        return BanditDecision(
            selected_algorithm=selected_algorithm,
            selection_probability=1.0,  # Rule-based is deterministic
            confidence_score=0.8,       # High confidence in rules
            context_match_score=1.0,    # Perfect rule match
            exploration_factor=0.0,     # No exploration in rule-based
            decision_rationale=rationale
        )


def create_router_guard() -> RouterGuard:
    """Create a RouterGuard instance."""
    return RouterGuard()
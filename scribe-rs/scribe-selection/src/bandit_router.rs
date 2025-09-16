//! Bandit Router System for V5 Variant
//!
//! Implements multi-armed bandit algorithms for intelligent selection strategy routing.
//! Uses Thompson Sampling and Upper Confidence Bound (UCB) algorithms to optimize
//! selection performance across different contexts and file types.

use scribe_core::{Result, ScribeError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::E;

/// Configuration for the bandit router system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditConfig {
    /// Exploration parameter for UCB algorithm
    pub exploration_factor: f64,
    /// Minimum trials before using bandit decisions
    pub min_trials: usize,
    /// Decay factor for reward history (0.0-1.0)
    pub decay_factor: f64,
    /// Maximum context window for learning
    pub context_window: usize,
    /// Enable Thompson Sampling (vs UCB)
    pub use_thompson_sampling: bool,
}

impl Default for BanditConfig {
    fn default() -> Self {
        Self {
            exploration_factor: 1.414, // sqrt(2) for UCB
            min_trials: 10,
            decay_factor: 0.95,
            context_window: 1000,
            use_thompson_sampling: true,
        }
    }
}

/// Selection strategy that can be routed
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Importance-based greedy selection
    ImportanceGreedy,
    /// Dependency-aware selection
    DependencyAware,
    /// Coverage-optimizing selection
    CoverageOptimized,
    /// Random selection (baseline)
    Random,
    /// Two-pass speculative selection
    TwoPassSpeculative,
    /// Quota-managed selection
    QuotaManaged,
}

impl SelectionStrategy {
    pub fn all_strategies() -> Vec<Self> {
        vec![
            Self::ImportanceGreedy,
            Self::DependencyAware,
            Self::CoverageOptimized,
            Self::Random,
            Self::TwoPassSpeculative,
            Self::QuotaManaged,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::ImportanceGreedy => "importance_greedy",
            Self::DependencyAware => "dependency_aware",
            Self::CoverageOptimized => "coverage_optimized",
            Self::Random => "random",
            Self::TwoPassSpeculative => "two_pass_speculative",
            Self::QuotaManaged => "quota_managed",
        }
    }
}

/// Context features for bandit decision making
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionContext {
    /// Number of available files
    pub file_count: usize,
    /// Average file importance score
    pub avg_importance: f64,
    /// Dependency graph density (edges/nodes)
    pub dependency_density: f64,
    /// Budget constraint ratio (available/total)
    pub budget_ratio: f64,
    /// Dominant file type (source, test, config, etc.)
    pub dominant_file_type: String,
    /// Project size category (small, medium, large)
    pub project_size: ProjectSize,
    /// Time constraint (tight, normal, relaxed)
    pub time_constraint: TimeConstraint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProjectSize {
    Small,  // < 50 files
    Medium, // 50-500 files
    Large,  // > 500 files
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeConstraint {
    Tight,   // Need results quickly
    Normal,  // Standard time expectations
    Relaxed, // Can afford thorough analysis
}

/// Bandit arm statistics for tracking performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmStats {
    /// Strategy this arm represents
    pub strategy: SelectionStrategy,
    /// Number of times this arm was selected
    pub trials: usize,
    /// Sum of rewards received
    pub total_reward: f64,
    /// Average reward (total_reward / trials)
    pub avg_reward: f64,
    /// For Thompson Sampling: Beta distribution parameters
    pub alpha: f64, // Successes + 1
    pub beta: f64, // Failures + 1
    /// Recent reward history for decay calculation
    pub recent_rewards: Vec<f64>,
}

impl ArmStats {
    pub fn new(strategy: SelectionStrategy) -> Self {
        Self {
            strategy,
            trials: 0,
            total_reward: 0.0,
            avg_reward: 0.0,
            alpha: 1.0, // Uniform prior
            beta: 1.0,  // Uniform prior
            recent_rewards: Vec::new(),
        }
    }

    /// Update statistics with a new reward
    pub fn update(&mut self, reward: f64, decay_factor: f64, max_history: usize) {
        self.trials += 1;
        self.total_reward += reward;
        self.avg_reward = self.total_reward / self.trials as f64;

        // Update Beta distribution parameters for Thompson Sampling
        if reward > 0.5 {
            // Consider > 0.5 as success
            self.alpha += reward;
        } else {
            self.beta += 1.0 - reward;
        }

        // Maintain recent rewards with decay
        self.recent_rewards.push(reward);
        if self.recent_rewards.len() > max_history {
            self.recent_rewards.remove(0);
        }

        // Apply decay to older rewards
        if decay_factor < 1.0 {
            for i in 0..self.recent_rewards.len() {
                let age = self.recent_rewards.len() - i - 1;
                self.recent_rewards[i] *= decay_factor.powi(age as i32);
            }
        }
    }

    /// Calculate Upper Confidence Bound
    pub fn ucb_score(&self, total_trials: usize, exploration_factor: f64) -> f64 {
        if self.trials == 0 {
            f64::INFINITY // Explore unvisited arms first
        } else {
            let confidence_interval =
                exploration_factor * ((total_trials as f64).ln() / self.trials as f64).sqrt();
            self.avg_reward + confidence_interval
        }
    }

    /// Sample from Beta distribution for Thompson Sampling
    pub fn thompson_sample(&self) -> f64 {
        if self.trials == 0 {
            0.5 // Uniform prior sample
        } else {
            // Simplified Beta sampling using ratio of uniforms
            let x = self.sample_gamma(self.alpha);
            let y = self.sample_gamma(self.beta);
            x / (x + y)
        }
    }

    /// Simple gamma distribution sampling (approximation)
    fn sample_gamma(&self, shape: f64) -> f64 {
        if shape < 1.0 {
            // Use rejection sampling for shape < 1
            let mut rng = fastrand::Rng::new();
            loop {
                let u = rng.f64();
                let v = rng.f64();
                let x = shape.powf(1.0 / shape) * u.powf(1.0 / shape);
                if v <= (-x).exp() {
                    return x;
                }
            }
        } else {
            // Marsaglia and Tsang's method for shape >= 1
            let d = shape - 1.0 / 3.0;
            let c = 1.0 / (9.0 * d).sqrt();
            let mut rng = fastrand::Rng::new();

            loop {
                let x = rng.f64();
                let v = 1.0 + c * x;
                if v > 0.0 {
                    let v = v * v * v;
                    let u = rng.f64();
                    if u < 1.0 - 0.0331 * x * x * x * x {
                        return d * v;
                    }
                    if (u).ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                        return d * v;
                    }
                }
            }
        }
    }
}

/// Result of bandit routing decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// Selected strategy
    pub strategy: SelectionStrategy,
    /// Confidence in this decision (0.0-1.0)
    pub confidence: f64,
    /// Exploration vs exploitation (true = exploration)
    pub is_exploration: bool,
    /// All strategy scores for debugging
    pub strategy_scores: HashMap<SelectionStrategy, f64>,
    /// Context that influenced the decision
    pub context_features: SelectionContext,
}

/// Feedback about the performance of a routing decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceFeedback {
    /// Strategy that was used
    pub strategy: SelectionStrategy,
    /// Quality score achieved (0.0-1.0)
    pub quality_score: f64,
    /// Budget utilization efficiency (0.0-1.0)
    pub budget_efficiency: f64,
    /// User satisfaction score if available (0.0-1.0)
    pub user_satisfaction: Option<f64>,
    /// Time taken relative to budget (0.0-1.0)
    pub time_efficiency: f64,
    /// Context this feedback applies to
    pub context: SelectionContext,
}

impl PerformanceFeedback {
    /// Calculate overall reward from individual metrics
    pub fn calculate_reward(&self) -> f64 {
        let mut reward = 0.0;
        let mut weight_sum = 0.0;

        // Quality is most important (40% weight)
        reward += self.quality_score * 0.4;
        weight_sum += 0.4;

        // Budget efficiency is important (30% weight)
        reward += self.budget_efficiency * 0.3;
        weight_sum += 0.3;

        // Time efficiency matters (20% weight)
        reward += self.time_efficiency * 0.2;
        weight_sum += 0.2;

        // User satisfaction if available (10% weight)
        if let Some(satisfaction) = self.user_satisfaction {
            reward += satisfaction * 0.1;
            weight_sum += 0.1;
        }

        reward / weight_sum
    }
}

/// Multi-armed bandit router for selection strategies
pub struct BanditRouter {
    config: BanditConfig,
    arms: HashMap<SelectionStrategy, ArmStats>,
    total_trials: usize,
    context_history: Vec<(SelectionContext, SelectionStrategy, f64)>,
}

impl BanditRouter {
    /// Create new bandit router with default configuration
    pub fn new() -> Self {
        Self::with_config(BanditConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: BanditConfig) -> Self {
        let mut arms = HashMap::new();
        for strategy in SelectionStrategy::all_strategies() {
            arms.insert(strategy.clone(), ArmStats::new(strategy));
        }

        Self {
            config,
            arms,
            total_trials: 0,
            context_history: Vec::new(),
        }
    }

    /// Route a selection request to the best strategy
    pub fn route_selection(&mut self, context: SelectionContext) -> Result<RoutingDecision> {
        let mut strategy_scores = HashMap::new();
        let mut best_strategy = SelectionStrategy::ImportanceGreedy;
        let mut best_score = f64::NEG_INFINITY;
        let mut is_exploration = false;

        // If we don't have enough trials, explore uniformly
        if self.total_trials < self.config.min_trials {
            let strategies = SelectionStrategy::all_strategies();
            let idx = fastrand::usize(0..strategies.len());
            best_strategy = strategies[idx].clone();
            is_exploration = true;

            for strategy in &strategies {
                strategy_scores.insert(strategy.clone(), 0.5);
            }
        } else if self.config.use_thompson_sampling {
            // Thompson Sampling
            for (strategy, arm) in &self.arms {
                let score = arm.thompson_sample();
                strategy_scores.insert(strategy.clone(), score);

                if score > best_score {
                    best_score = score;
                    best_strategy = strategy.clone();
                }
            }
        } else {
            // Upper Confidence Bound (UCB)
            for (strategy, arm) in &self.arms {
                let score = arm.ucb_score(self.total_trials, self.config.exploration_factor);
                strategy_scores.insert(strategy.clone(), score);

                if score > best_score {
                    best_score = score;
                    best_strategy = strategy.clone();
                    is_exploration = arm.trials < self.config.min_trials;
                }
            }
        }

        // Apply contextual adjustments
        best_strategy = self.apply_contextual_adjustments(&context, best_strategy);

        // Calculate confidence based on arm statistics
        let confidence = if let Some(arm) = self.arms.get(&best_strategy) {
            if arm.trials > 0 {
                // Higher confidence with more trials and consistent performance
                let trial_confidence = (arm.trials as f64 / 100.0).min(1.0);
                let performance_confidence = arm.avg_reward;
                (trial_confidence + performance_confidence) / 2.0
            } else {
                0.1 // Low confidence for unexplored arms
            }
        } else {
            0.0
        };

        Ok(RoutingDecision {
            strategy: best_strategy,
            confidence,
            is_exploration,
            strategy_scores,
            context_features: context,
        })
    }

    /// Apply contextual heuristics to adjust strategy selection
    fn apply_contextual_adjustments(
        &self,
        context: &SelectionContext,
        base_strategy: SelectionStrategy,
    ) -> SelectionStrategy {
        // For very small projects, simple strategies work well
        if matches!(context.project_size, ProjectSize::Small) && context.file_count < 20 {
            return SelectionStrategy::ImportanceGreedy;
        }

        // For high dependency density, dependency-aware strategy is better
        if context.dependency_density > 0.7 {
            return SelectionStrategy::DependencyAware;
        }

        // For tight time constraints, avoid complex strategies
        if matches!(context.time_constraint, TimeConstraint::Tight) {
            return match base_strategy {
                SelectionStrategy::TwoPassSpeculative => SelectionStrategy::ImportanceGreedy,
                SelectionStrategy::CoverageOptimized => SelectionStrategy::DependencyAware,
                _ => base_strategy,
            };
        }

        // For low budget ratios, quota management is essential
        if context.budget_ratio < 0.3 {
            return SelectionStrategy::QuotaManaged;
        }

        base_strategy
    }

    /// Provide feedback on strategy performance
    pub fn provide_feedback(&mut self, feedback: PerformanceFeedback) -> Result<()> {
        let reward = feedback.calculate_reward();

        if let Some(arm) = self.arms.get_mut(&feedback.strategy) {
            arm.update(reward, self.config.decay_factor, self.config.context_window);
            self.total_trials += 1;

            // Store context history for learning
            self.context_history.push((
                feedback.context.clone(),
                feedback.strategy.clone(),
                reward,
            ));

            // Maintain context window
            if self.context_history.len() > self.config.context_window {
                self.context_history.remove(0);
            }
        }

        Ok(())
    }

    /// Get current performance statistics
    pub fn get_statistics(&self) -> BanditStatistics {
        let mut strategy_stats = HashMap::new();

        for (strategy, arm) in &self.arms {
            strategy_stats.insert(strategy.clone(), arm.clone());
        }

        BanditStatistics {
            total_trials: self.total_trials,
            strategy_performance: strategy_stats,
            best_strategy: self.get_best_strategy(),
            exploration_rate: self.calculate_exploration_rate(),
        }
    }

    /// Get the currently best performing strategy
    fn get_best_strategy(&self) -> Option<SelectionStrategy> {
        self.arms
            .iter()
            .filter(|(_, arm)| arm.trials > 0)
            .max_by(|(_, a), (_, b)| a.avg_reward.partial_cmp(&b.avg_reward).unwrap())
            .map(|(strategy, _)| strategy.clone())
    }

    /// Calculate current exploration rate
    fn calculate_exploration_rate(&self) -> f64 {
        let recent_trials = self.context_history.len().min(50);
        if recent_trials == 0 {
            return 1.0;
        }

        let mut exploration_count = 0;
        for (_, strategy, _) in &self.context_history {
            if let Some(arm) = self.arms.get(strategy) {
                if arm.trials < self.config.min_trials {
                    exploration_count += 1;
                }
            }
        }

        exploration_count as f64 / recent_trials as f64
    }

    /// Reset all statistics (for testing or retraining)
    pub fn reset(&mut self) {
        for (_, arm) in &mut self.arms {
            *arm = ArmStats::new(arm.strategy.clone());
        }
        self.total_trials = 0;
        self.context_history.clear();
    }

    /// Export model state for persistence
    pub fn export_state(&self) -> BanditState {
        BanditState {
            config: self.config.clone(),
            arms: self.arms.clone(),
            total_trials: self.total_trials,
            context_history: self.context_history.clone(),
        }
    }

    /// Import model state from persistence
    pub fn import_state(&mut self, state: BanditState) -> Result<()> {
        self.config = state.config;
        self.arms = state.arms;
        self.total_trials = state.total_trials;
        self.context_history = state.context_history;
        Ok(())
    }
}

/// Serializable bandit router state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditState {
    pub config: BanditConfig,
    pub arms: HashMap<SelectionStrategy, ArmStats>,
    pub total_trials: usize,
    pub context_history: Vec<(SelectionContext, SelectionStrategy, f64)>,
}

/// Performance statistics for the bandit router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditStatistics {
    /// Total number of trials across all arms
    pub total_trials: usize,
    /// Per-strategy performance statistics
    pub strategy_performance: HashMap<SelectionStrategy, ArmStats>,
    /// Currently best performing strategy
    pub best_strategy: Option<SelectionStrategy>,
    /// Current exploration rate (0.0-1.0)
    pub exploration_rate: f64,
}

impl Default for BanditRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_context() -> SelectionContext {
        SelectionContext {
            file_count: 100,
            avg_importance: 0.7,
            dependency_density: 0.5,
            budget_ratio: 0.8,
            dominant_file_type: "source".to_string(),
            project_size: ProjectSize::Medium,
            time_constraint: TimeConstraint::Normal,
        }
    }

    fn create_test_feedback(strategy: SelectionStrategy, quality: f64) -> PerformanceFeedback {
        PerformanceFeedback {
            strategy,
            quality_score: quality,
            budget_efficiency: 0.8,
            user_satisfaction: Some(0.7),
            time_efficiency: 0.9,
            context: create_test_context(),
        }
    }

    #[test]
    fn test_bandit_router_creation() {
        let router = BanditRouter::new();
        assert_eq!(router.arms.len(), 6); // All strategies
        assert_eq!(router.total_trials, 0);
    }

    #[test]
    fn test_initial_routing_exploration() {
        let mut router = BanditRouter::new();
        let context = create_test_context();

        let decision = router.route_selection(context).unwrap();
        assert!(decision.is_exploration); // Should explore initially
        assert!(decision.confidence < 0.5); // Low confidence initially
    }

    #[test]
    fn test_feedback_learning() {
        let mut router = BanditRouter::new();

        // Provide positive feedback for ImportanceGreedy
        let feedback = create_test_feedback(SelectionStrategy::ImportanceGreedy, 0.9);
        router.provide_feedback(feedback).unwrap();

        // Check that statistics updated
        let stats = router.get_statistics();
        let greedy_stats = stats
            .strategy_performance
            .get(&SelectionStrategy::ImportanceGreedy)
            .unwrap();
        assert_eq!(greedy_stats.trials, 1);
        assert!(greedy_stats.avg_reward > 0.0);
    }

    #[test]
    fn test_strategy_selection_improvement() {
        let mut router = BanditRouter::new();
        let context = create_test_context();

        // Train the router with consistent feedback
        for _ in 0..20 {
            // ImportanceGreedy performs well
            let feedback = create_test_feedback(SelectionStrategy::ImportanceGreedy, 0.9);
            router.provide_feedback(feedback).unwrap();

            // Random performs poorly
            let feedback = create_test_feedback(SelectionStrategy::Random, 0.3);
            router.provide_feedback(feedback).unwrap();
        }

        // After training, should prefer ImportanceGreedy
        let decision = router.route_selection(context).unwrap();
        assert!(!decision.is_exploration); // Should exploit now

        let stats = router.get_statistics();
        assert_eq!(
            stats.best_strategy,
            Some(SelectionStrategy::ImportanceGreedy)
        );
    }

    #[test]
    fn test_contextual_adjustments() {
        let router = BanditRouter::new();

        // Small project should prefer simple strategy
        let mut small_context = create_test_context();
        small_context.project_size = ProjectSize::Small;
        small_context.file_count = 15;

        let adjusted = router
            .apply_contextual_adjustments(&small_context, SelectionStrategy::TwoPassSpeculative);
        assert_eq!(adjusted, SelectionStrategy::ImportanceGreedy);

        // High dependency density should prefer dependency-aware
        let mut dep_context = create_test_context();
        dep_context.dependency_density = 0.8;

        let adjusted = router.apply_contextual_adjustments(&dep_context, SelectionStrategy::Random);
        assert_eq!(adjusted, SelectionStrategy::DependencyAware);
    }

    #[test]
    fn test_arm_statistics() {
        let mut arm = ArmStats::new(SelectionStrategy::ImportanceGreedy);

        // Initial state
        assert_eq!(arm.trials, 0);
        assert_eq!(arm.avg_reward, 0.0);

        // Update with rewards
        arm.update(0.8, 0.95, 100);
        arm.update(0.9, 0.95, 100);

        assert_eq!(arm.trials, 2);
        assert!((arm.avg_reward - 0.85).abs() < 0.01);

        // UCB score should be high for good performance
        let ucb = arm.ucb_score(10, 1.414);
        assert!(ucb > arm.avg_reward);
    }

    #[test]
    fn test_thompson_sampling() {
        let mut arm = ArmStats::new(SelectionStrategy::ImportanceGreedy);

        // Add some positive rewards
        for _ in 0..10 {
            arm.update(0.8, 0.95, 100);
        }

        // Thompson samples should be reasonable
        let sample = arm.thompson_sample();
        assert!(sample > 0.0);
        assert!(sample < 1.0);

        // Multiple samples should vary
        let samples: Vec<f64> = (0..10).map(|_| arm.thompson_sample()).collect();
        let variance = samples
            .iter()
            .map(|s| (s - arm.avg_reward).powi(2))
            .sum::<f64>()
            / 10.0;
        assert!(variance > 0.0); // Should have some variance
    }

    #[test]
    fn test_performance_feedback_reward_calculation() {
        let feedback = PerformanceFeedback {
            strategy: SelectionStrategy::ImportanceGreedy,
            quality_score: 0.9,
            budget_efficiency: 0.8,
            user_satisfaction: Some(0.7),
            time_efficiency: 0.85,
            context: create_test_context(),
        };

        let reward = feedback.calculate_reward();
        assert!(reward > 0.7); // Should be high with good metrics
        assert!(reward < 1.0);

        // Test without user satisfaction
        let feedback_no_user = PerformanceFeedback {
            user_satisfaction: None,
            ..feedback
        };

        let reward_no_user = feedback_no_user.calculate_reward();
        assert!(reward_no_user > 0.7);
    }

    #[test]
    fn test_state_persistence() {
        let mut router = BanditRouter::new();

        // Train the router
        let feedback = create_test_feedback(SelectionStrategy::ImportanceGreedy, 0.9);
        router.provide_feedback(feedback).unwrap();

        // Export and import state
        let state = router.export_state();
        let mut new_router = BanditRouter::new();
        new_router.import_state(state).unwrap();

        // Should have same statistics
        assert_eq!(router.total_trials, new_router.total_trials);
        assert_eq!(router.arms.len(), new_router.arms.len());
    }
}

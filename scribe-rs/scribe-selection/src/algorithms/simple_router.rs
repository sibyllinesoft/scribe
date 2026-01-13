//! Simple rule-based router for file selection strategy selection.
//!
//! Replaces the complex multi-armed bandit approach with a straightforward
//! decision tree based on project context and constraints.

use serde::{Deserialize, Serialize};

/// Selection strategy options
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

/// Project size categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProjectSize {
    Small,  // < 50 files
    Medium, // 50-500 files
    Large,  // > 500 files
}

/// Time constraint levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeConstraint {
    Tight,   // Need results quickly
    Normal,  // Standard time expectations
    Relaxed, // Can afford thorough analysis
}

/// Context features for routing decisions
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
    /// Project size category
    pub project_size: ProjectSize,
    /// Time constraint level
    pub time_constraint: TimeConstraint,
}

/// Result of a routing decision
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Selected strategy
    pub strategy: SelectionStrategy,
    /// Reason for this choice (for debugging/logging)
    pub reason: String,
}

/// Simple rule-based router for selection strategy
pub struct SimpleRouter;

impl SimpleRouter {
    /// Create a new simple router
    pub fn new() -> Self {
        Self
    }

    /// Route to the appropriate selection strategy based on context
    ///
    /// Decision priority:
    /// 1. Budget constraints (critical)
    /// 2. Time constraints (important)
    /// 3. Project size and complexity (characteristics)
    /// 4. Default to balanced approach
    pub fn route_selection(&self, context: &SelectionContext) -> RoutingDecision {
        // Priority 1: Budget constraints are critical
        if context.budget_ratio < 0.3 {
            return RoutingDecision {
                strategy: SelectionStrategy::QuotaManaged,
                reason: format!(
                    "Low budget ratio ({:.2}) requires quota management",
                    context.budget_ratio
                ),
            };
        }

        // Priority 2: Time constraints affect strategy complexity
        if matches!(context.time_constraint, TimeConstraint::Tight) {
            // For tight deadlines with complex dependencies, use dependency-aware
            if context.dependency_density > 0.5 {
                return RoutingDecision {
                    strategy: SelectionStrategy::DependencyAware,
                    reason: format!(
                        "Tight time constraint with high dependency density ({:.2}) needs dependency-aware selection",
                        context.dependency_density
                    ),
                };
            }
            // Otherwise use the fastest simple strategy
            return RoutingDecision {
                strategy: SelectionStrategy::ImportanceGreedy,
                reason: "Tight time constraint requires fast importance-based selection".to_string(),
            };
        }

        // Priority 3: Small projects benefit from simple strategies
        if matches!(context.project_size, ProjectSize::Small) {
            return RoutingDecision {
                strategy: SelectionStrategy::ImportanceGreedy,
                reason: format!(
                    "Small project ({} files) works well with importance-based selection",
                    context.file_count
                ),
            };
        }

        // Priority 4: High dependency density needs specialized handling
        if context.dependency_density > 0.7 {
            return RoutingDecision {
                strategy: SelectionStrategy::DependencyAware,
                reason: format!(
                    "High dependency density ({:.2}) requires dependency-aware selection",
                    context.dependency_density
                ),
            };
        }

        // Priority 5: Large projects benefit from two-pass analysis
        if context.file_count > 200 {
            return RoutingDecision {
                strategy: SelectionStrategy::TwoPassSpeculative,
                reason: format!(
                    "Large project ({} files) benefits from two-pass speculative selection",
                    context.file_count
                ),
            };
        }

        // Default: Use coverage-optimized for balanced results
        RoutingDecision {
            strategy: SelectionStrategy::CoverageOptimized,
            reason: "Standard context: using coverage-optimized selection for balanced results".to_string(),
        }
    }
}

impl Default for SimpleRouter {
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
            avg_importance: 0.5,
            dependency_density: 0.3,
            budget_ratio: 0.8,
            dominant_file_type: "source".to_string(),
            project_size: ProjectSize::Medium,
            time_constraint: TimeConstraint::Normal,
        }
    }

    #[test]
    fn test_low_budget_uses_quota_managed() {
        let router = SimpleRouter::new();
        let mut context = create_test_context();
        context.budget_ratio = 0.2;

        let decision = router.route_selection(&context);
        assert_eq!(decision.strategy, SelectionStrategy::QuotaManaged);
        assert!(decision.reason.contains("budget"));
    }

    #[test]
    fn test_tight_time_with_dependencies_uses_dependency_aware() {
        let router = SimpleRouter::new();
        let mut context = create_test_context();
        context.time_constraint = TimeConstraint::Tight;
        context.dependency_density = 0.6;

        let decision = router.route_selection(&context);
        assert_eq!(decision.strategy, SelectionStrategy::DependencyAware);
    }

    #[test]
    fn test_tight_time_without_dependencies_uses_importance_greedy() {
        let router = SimpleRouter::new();
        let mut context = create_test_context();
        context.time_constraint = TimeConstraint::Tight;
        context.dependency_density = 0.2;

        let decision = router.route_selection(&context);
        assert_eq!(decision.strategy, SelectionStrategy::ImportanceGreedy);
    }

    #[test]
    fn test_small_project_uses_importance_greedy() {
        let router = SimpleRouter::new();
        let mut context = create_test_context();
        context.project_size = ProjectSize::Small;
        context.file_count = 30;

        let decision = router.route_selection(&context);
        assert_eq!(decision.strategy, SelectionStrategy::ImportanceGreedy);
    }

    #[test]
    fn test_high_dependency_density_uses_dependency_aware() {
        let router = SimpleRouter::new();
        let mut context = create_test_context();
        context.dependency_density = 0.8;

        let decision = router.route_selection(&context);
        assert_eq!(decision.strategy, SelectionStrategy::DependencyAware);
    }

    #[test]
    fn test_large_file_count_uses_two_pass() {
        let router = SimpleRouter::new();
        let mut context = create_test_context();
        context.file_count = 250;

        let decision = router.route_selection(&context);
        assert_eq!(decision.strategy, SelectionStrategy::TwoPassSpeculative);
    }

    #[test]
    fn test_default_uses_coverage_optimized() {
        let router = SimpleRouter::new();
        let context = create_test_context();

        let decision = router.route_selection(&context);
        assert_eq!(decision.strategy, SelectionStrategy::CoverageOptimized);
    }

    #[test]
    fn test_priority_order_budget_over_time() {
        let router = SimpleRouter::new();
        let mut context = create_test_context();
        context.budget_ratio = 0.2;
        context.time_constraint = TimeConstraint::Tight;

        let decision = router.route_selection(&context);
        // Budget constraint should take priority
        assert_eq!(decision.strategy, SelectionStrategy::QuotaManaged);
    }

    #[test]
    fn test_priority_order_time_over_size() {
        let router = SimpleRouter::new();
        let mut context = create_test_context();
        context.time_constraint = TimeConstraint::Tight;
        context.project_size = ProjectSize::Small;
        context.dependency_density = 0.2;

        let decision = router.route_selection(&context);
        // Time constraint should take priority over project size
        assert_eq!(decision.strategy, SelectionStrategy::ImportanceGreedy);
    }

    #[test]
    fn test_selection_strategy_names() {
        assert_eq!(SelectionStrategy::ImportanceGreedy.name(), "importance_greedy");
        assert_eq!(SelectionStrategy::DependencyAware.name(), "dependency_aware");
        assert_eq!(SelectionStrategy::CoverageOptimized.name(), "coverage_optimized");
        assert_eq!(SelectionStrategy::Random.name(), "random");
        assert_eq!(SelectionStrategy::TwoPassSpeculative.name(), "two_pass_speculative");
        assert_eq!(SelectionStrategy::QuotaManaged.name(), "quota_managed");
    }

    #[test]
    fn test_router_default() {
        let router = SimpleRouter::default();
        let context = create_test_context();
        // Should work just like new()
        let decision = router.route_selection(&context);
        assert_eq!(decision.strategy, SelectionStrategy::CoverageOptimized);
    }

    #[test]
    fn test_routing_decision_has_reason() {
        let router = SimpleRouter::new();
        let context = create_test_context();
        let decision = router.route_selection(&context);
        // All decisions should have a non-empty reason
        assert!(!decision.reason.is_empty());
    }
}

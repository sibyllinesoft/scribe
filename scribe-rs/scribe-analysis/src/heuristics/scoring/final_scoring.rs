//! Final score calculation and combination logic

use super::normalization::NormalizedScores;
use super::types::{HeuristicWeights, ScoreComponents};

/// Calculate the final weighted score from normalized components
pub fn calculate_final_score(
    normalized_scores: &NormalizedScores,
    weights: &HeuristicWeights,
    template_boost: f64,
    priority_boost: f64,
) -> f64 {
    let weighted_sum = weights.doc_weight * normalized_scores.doc_score
        + weights.readme_weight * normalized_scores.readme_score
        + weights.import_weight * normalized_scores.import_score
        + weights.path_weight * normalized_scores.path_score
        + weights.test_link_weight * normalized_scores.test_link_score
        + weights.churn_weight * normalized_scores.churn_score
        + weights.centrality_weight * normalized_scores.centrality_score
        + weights.entrypoint_weight * normalized_scores.entrypoint_score
        + weights.examples_weight * normalized_scores.examples_score;

    weighted_sum + template_boost + priority_boost
}

impl ScoreComponents {
    /// Get the top contributing factors to this score
    pub fn top_factors(&self, count: usize) -> Vec<(String, f64)> {
        let mut factors = vec![
            (
                "doc_score".to_string(),
                self.doc_score * self.weights.doc_weight,
            ),
            (
                "readme_score".to_string(),
                self.readme_score * self.weights.readme_weight,
            ),
            (
                "import_score".to_string(),
                self.import_score * self.weights.import_weight,
            ),
            (
                "path_score".to_string(),
                self.path_score * self.weights.path_weight,
            ),
            (
                "test_link_score".to_string(),
                self.test_link_score * self.weights.test_link_weight,
            ),
            (
                "churn_score".to_string(),
                self.churn_score * self.weights.churn_weight,
            ),
            (
                "centrality_score".to_string(),
                self.centrality_score * self.weights.centrality_weight,
            ),
            (
                "entrypoint_score".to_string(),
                self.entrypoint_score * self.weights.entrypoint_weight,
            ),
            (
                "examples_score".to_string(),
                self.examples_score * self.weights.examples_weight,
            ),
            ("priority_boost".to_string(), self.priority_boost),
            ("template_boost".to_string(), self.template_boost),
        ];

        factors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        factors.truncate(count);
        factors
    }

    /// Check if this score represents a high-confidence result
    pub fn is_high_confidence(&self) -> bool {
        let contributing_factors = self
            .top_factors(11)
            .iter()
            .filter(|(_, score)| *score > 0.1)
            .count();

        // High confidence if multiple factors contribute significantly
        contributing_factors >= 3 && self.final_score > 1.0
    }

    /// Get a human-readable explanation of the score
    pub fn explanation(&self) -> String {
        let top_factors = self.top_factors(3);
        let mut explanation = format!("Score: {:.2}", self.final_score);

        if !top_factors.is_empty() {
            explanation.push_str(" (top factors: ");
            let factor_strings: Vec<String> = top_factors
                .iter()
                .map(|(name, score)| format!("{}: {:.2}", name.replace("_score", ""), score))
                .collect();
            explanation.push_str(&factor_strings.join(", "));
            explanation.push(')');
        }

        explanation
    }

    /// Determine the primary reason for this file's importance
    pub fn primary_importance_reason(&self) -> String {
        let top_factor = self.top_factors(1);
        if let Some((factor_name, score)) = top_factor.first() {
            match factor_name.as_str() {
                "doc_score" => format!("Documentation file (score: {:.2})", score),
                "readme_score" => format!("README file (score: {:.2})", score),
                "import_score" => format!("High import connectivity (score: {:.2})", score),
                "entrypoint_score" => format!("Application entrypoint (score: {:.2})", score),
                "centrality_score" => format!("Central to codebase (PageRank: {:.2})", score),
                "examples_score" => format!("Contains examples (score: {:.2})", score),
                "churn_score" => format!("Recently modified (score: {:.2})", score),
                "test_link_score" => format!("Test-related file (score: {:.2})", score),
                "priority_boost" => format!("Manually prioritized (boost: {:.2})", score),
                "template_boost" => format!("Template/config file (boost: {:.2})", score),
                _ => format!("Mixed factors (top: {:.2})", score),
            }
        } else {
            "No significant factors".to_string()
        }
    }

    /// Calculate a confidence interval for the score
    pub fn confidence_interval(&self) -> (f64, f64) {
        // Simple confidence estimation based on number of contributing factors
        let top_factors = self.top_factors(11);
        let significant_factors = top_factors.iter().filter(|(_, score)| *score > 0.1).count();

        let confidence_width = match significant_factors {
            0..=1 => 0.5, // Low confidence - wide interval
            2..=3 => 0.3, // Medium confidence
            4..=5 => 0.2, // High confidence
            _ => 0.1,     // Very high confidence - narrow interval
        };

        let lower = (self.final_score - confidence_width).max(0.0);
        let upper = self.final_score + confidence_width;

        (lower, upper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_final_score_calculation() {
        let normalized = NormalizedScores {
            doc_score: 1.0,
            readme_score: 0.5,
            import_score: 0.3,
            path_score: 0.7,
            test_link_score: 0.0,
            churn_score: 0.2,
            centrality_score: 0.8,
            entrypoint_score: 1.0,
            examples_score: 0.4,
        };

        let weights = HeuristicWeights::default();

        let final_score = calculate_final_score(&normalized, &weights, 0.5, 0.2);

        // Should be a positive weighted sum
        assert!(final_score > 0.0);

        // Should include boosts
        let base_score = calculate_final_score(&normalized, &weights, 0.0, 0.0);
        assert!(final_score > base_score);
    }

    #[test]
    fn test_score_explanation() {
        let score_components = ScoreComponents {
            final_score: 5.2,
            doc_score: 1.0,
            readme_score: 0.8,
            import_score: 0.5,
            path_score: 0.3,
            test_link_score: 0.0,
            churn_score: 0.1,
            centrality_score: 0.9,
            entrypoint_score: 1.0,
            examples_score: 0.4,
            priority_boost: 0.5,
            template_boost: 0.2,
            weights: HeuristicWeights::default(),
        };

        let explanation = score_components.explanation();
        assert!(explanation.contains("Score: 5.20"));
        assert!(explanation.contains("top factors"));

        let reason = score_components.primary_importance_reason();
        assert!(!reason.is_empty());

        let (lower, upper) = score_components.confidence_interval();
        assert!(lower <= score_components.final_score);
        assert!(upper >= score_components.final_score);
    }
}

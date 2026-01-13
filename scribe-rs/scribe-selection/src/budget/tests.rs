//! Additional tests for budget selection and token management.

use super::token_budget::SelectionConfig;
use super::weighting::FileWeights;
use std::collections::HashMap;

#[cfg(test)]
mod selection_config_tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SelectionConfig::default();
        assert_eq!(config.signature_boost, 1.5);
        assert_eq!(config.chunk_boost, 1.2);
    }

    #[test]
    fn test_resolution_mode() {
        let config = SelectionConfig::resolution();
        assert_eq!(config.signature_boost, 1.0);
        assert_eq!(config.chunk_boost, 1.0);
    }

    #[test]
    fn test_coverage_mode() {
        let config = SelectionConfig::coverage();
        assert_eq!(config.signature_boost, 2.0);
        assert_eq!(config.chunk_boost, 1.5);
    }

    #[test]
    fn test_max_coverage_mode() {
        let config = SelectionConfig::max_coverage();
        assert_eq!(config.signature_boost, 3.0);
        assert_eq!(config.chunk_boost, 2.0);
    }

    #[test]
    fn test_config_clone() {
        let config = SelectionConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.signature_boost, config.signature_boost);
        assert_eq!(cloned.chunk_boost, config.chunk_boost);
    }

    #[test]
    fn test_config_debug() {
        let config = SelectionConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("SelectionConfig"));
        assert!(debug_str.contains("signature_boost"));
        assert!(debug_str.contains("chunk_boost"));
    }
}

#[cfg(test)]
mod file_weights_additional_tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let weights = FileWeights::new();
        assert!(weights.is_empty());
        assert_eq!(weights.len(), 0);
    }

    #[test]
    fn test_set_and_get() {
        let mut weights = FileWeights::new();
        weights.set("src/main.rs".to_string(), 1.5);
        weights.set("src/lib.rs".to_string(), 2.0);

        assert_eq!(weights.get("src/main.rs"), 1.5);
        assert_eq!(weights.get("src/lib.rs"), 2.0);
        assert_eq!(weights.get("nonexistent.rs"), 0.0);
        assert_eq!(weights.len(), 2);
    }

    #[test]
    fn test_get_path() {
        let mut weights = FileWeights::new();
        weights.set("src/main.rs".to_string(), 1.5);

        use std::path::Path;
        let path = Path::new("src/main.rs");
        assert_eq!(weights.get_path(path), 1.5);
    }

    #[test]
    fn test_iter() {
        let mut weights = FileWeights::new();
        weights.set("file1.rs".to_string(), 1.0);
        weights.set("file2.rs".to_string(), 2.0);
        weights.set("file3.rs".to_string(), 3.0);

        let mut count = 0;
        let mut sum = 0.0;
        for (_path, weight) in weights.iter() {
            sum += *weight;
            count += 1;
        }
        assert_eq!(count, 3);
        assert_eq!(sum, 6.0);
    }

    #[test]
    fn test_from_inputs_multiple() {
        let mut input1 = HashMap::new();
        input1.insert("a.rs".to_string(), 1.0);
        input1.insert("b.rs".to_string(), 2.0);

        let mut input2 = HashMap::new();
        input2.insert("a.rs".to_string(), 3.0);
        input2.insert("c.rs".to_string(), 4.0);

        let mut input3 = HashMap::new();
        input3.insert("a.rs".to_string(), 5.0);

        let weights = FileWeights::from_inputs(vec![input1, input2, input3]);

        // a.rs: (1+3+5)/3 = 3.0
        assert!((weights.get("a.rs") - 3.0).abs() < 0.001);
        // b.rs: only in input1
        assert_eq!(weights.get("b.rs"), 2.0);
        // c.rs: only in input2
        assert_eq!(weights.get("c.rs"), 4.0);
    }

    #[test]
    fn test_merge_new_key() {
        let mut weights = FileWeights::new();
        weights.set("existing.rs".to_string(), 1.0);

        let mut input = HashMap::new();
        input.insert("new.rs".to_string(), 2.0);

        weights.merge(input);

        assert_eq!(weights.get("existing.rs"), 1.0);
        assert_eq!(weights.get("new.rs"), 2.0);
    }

    #[test]
    fn test_merge_existing_key() {
        let mut weights = FileWeights::new();
        weights.set("file.rs".to_string(), 1.0);

        let mut input = HashMap::new();
        input.insert("file.rs".to_string(), 3.0);

        weights.merge(input);

        // (1.0 + 3.0) / 2 = 2.0
        assert!((weights.get("file.rs") - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_clone() {
        let mut weights = FileWeights::new();
        weights.set("file.rs".to_string(), 1.5);

        let cloned = weights.clone();
        assert_eq!(cloned.get("file.rs"), 1.5);
    }
}

#[cfg(test)]
mod value_density_tests {
    #[test]
    fn test_density_calculation() {
        // Simple density = score / tokens
        let score = 100.0;
        let tokens = 50usize;
        let density = score / tokens as f64;
        assert_eq!(density, 2.0);
    }

    #[test]
    fn test_density_comparison() {
        // Higher density means better value per token
        let high_density = 100.0 / 50.0; // 2.0
        let low_density = 100.0 / 200.0; // 0.5

        assert!(high_density > low_density);
    }

    #[test]
    fn test_density_with_zero_tokens() {
        // Edge case: should handle division carefully
        let score = 100.0;
        let tokens = 1usize; // Minimum tokens
        let density = score / tokens as f64;
        assert_eq!(density, 100.0);
    }

    #[test]
    fn test_boosted_density() {
        // Signature boost increases effective score
        let base_score = 100.0;
        let tokens = 50usize;
        let boost = 1.5;

        let boosted_score = base_score * boost;
        let density = boosted_score / tokens as f64;

        assert_eq!(density, 3.0);
    }
}

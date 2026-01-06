//! File weighting layer for merging multiple weight inputs.
//!
//! This module accepts multiple weight maps from external sources and merges
//! them into a single weight per file by averaging. The merged weights are used
//! during selection to prioritize files for full content vs signature inclusion.

use std::collections::HashMap;
use std::path::Path;

/// Merged weights for files, computed by averaging multiple input sources.
#[derive(Debug, Clone, Default)]
pub struct FileWeights {
    /// file path -> merged weight
    weights: HashMap<String, f64>,
}

impl FileWeights {
    /// Create an empty weight set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create weights from multiple input maps, averaging where files overlap.
    ///
    /// Each input map is `file_path -> weight`. For files appearing in multiple
    /// inputs, the weights are averaged.
    pub fn from_inputs(inputs: Vec<HashMap<String, f64>>) -> Self {
        if inputs.is_empty() {
            return Self::new();
        }

        // Track sum and count for each file
        let mut sums: HashMap<String, f64> = HashMap::new();
        let mut counts: HashMap<String, usize> = HashMap::new();

        for input in inputs {
            for (path, weight) in input {
                *sums.entry(path.clone()).or_insert(0.0) += weight;
                *counts.entry(path).or_insert(0) += 1;
            }
        }

        // Compute averages
        let weights = sums
            .into_iter()
            .map(|(path, sum)| {
                let count = counts.get(&path).copied().unwrap_or(1);
                (path, sum / count as f64)
            })
            .collect();

        Self { weights }
    }

    /// Merge additional weights into this set by averaging with existing values.
    pub fn merge(&mut self, input: HashMap<String, f64>) {
        for (path, weight) in input {
            self.weights
                .entry(path)
                .and_modify(|existing| *existing = (*existing + weight) / 2.0)
                .or_insert(weight);
        }
    }

    /// Get the weight for a file path, returning 0.0 if not present.
    pub fn get(&self, path: &str) -> f64 {
        self.weights.get(path).copied().unwrap_or(0.0)
    }

    /// Get the weight for a file path (Path variant).
    pub fn get_path(&self, path: &Path) -> f64 {
        self.get(&path.to_string_lossy())
    }

    /// Set a weight directly for a file path.
    pub fn set(&mut self, path: String, weight: f64) {
        self.weights.insert(path, weight);
    }

    /// Check if any weights are present.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Number of files with weights.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Iterate over all weights.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &f64)> {
        self.weights.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_inputs() {
        let weights = FileWeights::from_inputs(vec![]);
        assert!(weights.is_empty());
        assert_eq!(weights.get("foo.rs"), 0.0);
    }

    #[test]
    fn test_single_input() {
        let mut input = HashMap::new();
        input.insert("src/lib.rs".to_string(), 0.8);
        input.insert("src/main.rs".to_string(), 0.6);

        let weights = FileWeights::from_inputs(vec![input]);
        assert_eq!(weights.get("src/lib.rs"), 0.8);
        assert_eq!(weights.get("src/main.rs"), 0.6);
        assert_eq!(weights.get("unknown.rs"), 0.0);
    }

    #[test]
    fn test_average_multiple_inputs() {
        let mut input1 = HashMap::new();
        input1.insert("src/lib.rs".to_string(), 0.8);
        input1.insert("src/main.rs".to_string(), 0.4);

        let mut input2 = HashMap::new();
        input2.insert("src/lib.rs".to_string(), 0.6);
        input2.insert("src/other.rs".to_string(), 0.5);

        let weights = FileWeights::from_inputs(vec![input1, input2]);

        // lib.rs appears in both: (0.8 + 0.6) / 2 = 0.7
        assert!((weights.get("src/lib.rs") - 0.7).abs() < 0.001);
        // main.rs only in first input
        assert_eq!(weights.get("src/main.rs"), 0.4);
        // other.rs only in second input
        assert_eq!(weights.get("src/other.rs"), 0.5);
    }

    #[test]
    fn test_merge() {
        let mut input1 = HashMap::new();
        input1.insert("src/lib.rs".to_string(), 0.8);

        let mut weights = FileWeights::from_inputs(vec![input1]);

        let mut input2 = HashMap::new();
        input2.insert("src/lib.rs".to_string(), 0.4);
        input2.insert("src/new.rs".to_string(), 0.5);

        weights.merge(input2);

        // lib.rs: (0.8 + 0.4) / 2 = 0.6
        assert!((weights.get("src/lib.rs") - 0.6).abs() < 0.001);
        assert_eq!(weights.get("src/new.rs"), 0.5);
    }
}

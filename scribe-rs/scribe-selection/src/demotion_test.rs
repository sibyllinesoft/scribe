#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_PYTHON_CODE: &str = r#"
"""Module docstring"""
import os
import sys
from typing import List

DEBUG = True

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        result = a + b
        self.history.append(('add', a, b, result))
        return result
    
    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        result = a * b
        self.history.append(('multiply', a, b, result))
        return result

def main():
    """Main function."""
    calc = Calculator()
    print(calc.add(2, 3))
    print(calc.multiply(4, 5))

if __name__ == "__main__":
    main()
"#;

    #[test]
    fn test_demotion_engine_creation() {
        let engine = DemotionEngine::new();
        // Should not panic
        drop(engine);
    }

    #[test]
    fn test_full_fidelity_mode() {
        let mut engine = DemotionEngine::new();
        let result = engine.apply_demotion(
            "test.py",
            SAMPLE_PYTHON_CODE,
            FidelityMode::Full,
            None,
        );

        assert_eq!(result.fidelity_mode, FidelityMode::Full);
        assert_eq!(result.content, SAMPLE_PYTHON_CODE);
        assert_eq!(result.compression_ratio, 1.0);
        assert_eq!(result.quality_score, 1.0);
        assert_eq!(result.chunks_kept, 1);
        assert_eq!(result.chunks_total, 1);
    }

    #[test]
    fn test_chunk_fidelity_mode() {
        let mut engine = DemotionEngine::new();
        let result = engine.apply_demotion(
            "test.py",
            SAMPLE_PYTHON_CODE,
            FidelityMode::Chunk,
            Some(100), // Low token limit to force selection
        );

        assert_eq!(result.fidelity_mode, FidelityMode::Chunk);
        assert!(result.compression_ratio < 1.0);
        assert!(result.quality_score > 0.0);
        assert!(result.chunks_kept < result.chunks_total);
        assert!(!result.content.is_empty());
        
        // Should preserve important chunks like imports and class definitions
        assert!(result.content.contains("import") || result.content.contains("class"));
    }

    #[test]
    fn test_signature_fidelity_mode() {
        let mut engine = DemotionEngine::new();
        let result = engine.apply_demotion(
            "test.py",
            SAMPLE_PYTHON_CODE,
            FidelityMode::Signature,
            None,
        );

        assert_eq!(result.fidelity_mode, FidelityMode::Signature);
        assert!(result.compression_ratio < 1.0);
        assert!(result.quality_score > 0.0);
        assert!(!result.content.is_empty());
        
        // Should preserve signatures without method bodies
        assert!(result.content.contains("def add"));
        assert!(result.content.contains("class Calculator"));
        // Should not contain implementation details
        assert!(!result.content.contains("result = a + b"));
    }

    #[test]
    fn test_code_chunker() {
        let mut chunker = CodeChunker::new();
        let chunks = chunker.chunk_content(SAMPLE_PYTHON_CODE, "test.py");

        assert!(!chunks.is_empty());
        
        // Should identify different chunk types
        let chunk_types: Vec<&str> = chunks.iter().map(|c| c.chunk_type.as_str()).collect();
        assert!(chunk_types.contains(&"import"));
        assert!(chunk_types.contains(&"class"));
        assert!(chunk_types.contains(&"function"));
        
        // Should have importance scores
        for chunk in &chunks {
            assert!(chunk.importance_score >= 0.0);
            assert!(chunk.importance_score <= 1.0);
        }
        
        // Imports should have high importance
        let import_chunks: Vec<_> = chunks.iter().filter(|c| c.chunk_type == "import").collect();
        for chunk in import_chunks {
            assert!(chunk.importance_score >= 0.8);
        }
    }

    #[test]
    fn test_signature_extractor() {
        let mut extractor = SignatureExtractor::new();
        let signatures = extractor.extract_signatures(SAMPLE_PYTHON_CODE, "test.py");

        // Should contain function and class signatures
        assert!(signatures.contains("def add(self, a: int, b: int) -> int:"));
        assert!(signatures.contains("class Calculator:"));
        assert!(signatures.contains("import os"));
        
        // Should not contain implementation details
        assert!(!signatures.contains("result = a + b"));
        assert!(!signatures.contains("self.history.append"));
    }

    #[test]
    fn test_language_detection() {
        let mut chunker = CodeChunker::new();
        
        assert_eq!(chunker.detect_language("test.py"), Some("python".to_string()));
        assert_eq!(chunker.detect_language("test.js"), Some("javascript".to_string()));
        assert_eq!(chunker.detect_language("test.ts"), Some("typescript".to_string()));
        assert_eq!(chunker.detect_language("test.go"), Some("go".to_string()));
        assert_eq!(chunker.detect_language("test.rs"), Some("rust".to_string()));
        assert_eq!(chunker.detect_language("test.unknown"), None);
    }

    const SAMPLE_RUST_CODE: &str = r#"
use std::collections::HashMap;

pub const MAX_SIZE: usize = 1000;

/// A sample struct
pub struct DataProcessor {
    data: HashMap<String, i32>,
}

impl DataProcessor {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
    
    pub fn process(&mut self, key: String, value: i32) -> Option<i32> {
        self.data.insert(key, value)
    }
}

pub trait Processor {
    fn process_data(&self, input: &str) -> String;
}
"#;

    #[test]
    fn test_rust_code_chunking() {
        let mut chunker = CodeChunker::new();
        let chunks = chunker.chunk_content(SAMPLE_RUST_CODE, "test.rs");

        assert!(!chunks.is_empty());
        
        let chunk_types: Vec<&str> = chunks.iter().map(|c| c.chunk_type.as_str()).collect();
        assert!(chunk_types.contains(&"use"));
        assert!(chunk_types.contains(&"struct"));
        assert!(chunk_types.contains(&"impl"));
        assert!(chunk_types.contains(&"trait"));
    }

    #[test]
    fn test_rust_signature_extraction() {
        let mut extractor = SignatureExtractor::new();
        let signatures = extractor.extract_signatures(SAMPLE_RUST_CODE, "test.rs");

        assert!(signatures.contains("use std::collections::HashMap"));
        assert!(signatures.contains("pub struct DataProcessor"));
        assert!(signatures.contains("pub trait Processor"));
        assert!(signatures.contains("pub fn new() -> Self"));
        
        // Should not contain implementation
        assert!(!signatures.contains("HashMap::new()"));
        assert!(!signatures.contains("self.data.insert"));
    }

    #[test]
    fn test_quality_score_calculation() {
        let engine = DemotionEngine::new();
        
        let all_chunks = vec![
            ChunkInfo {
                start_line: 1,
                end_line: 2,
                chunk_type: "import".to_string(),
                content: "import os".to_string(),
                importance_score: 0.9,
                estimated_tokens: 10,
                dependencies: vec![],
            },
            ChunkInfo {
                start_line: 3,
                end_line: 10,
                chunk_type: "class".to_string(),
                content: "class Test: pass".to_string(),
                importance_score: 0.8,
                estimated_tokens: 20,
                dependencies: vec![],
            },
        ];

        let selected_chunks = vec![all_chunks[0].clone()]; // Only the import

        let quality = engine.calculate_quality_score(&selected_chunks, &all_chunks);
        
        // Should be 0.9 / (0.9 + 0.8) = 0.529...
        assert!((quality - 0.529).abs() < 0.01);
    }
}
//! # Tokenization Module
//!
//! This module provides accurate token counting using OpenAI's tiktoken tokenizer,
//! replacing the simple character-based estimation used previously.
//!
//! ## Features
//!
//! - **Accurate Token Counting**: Uses tiktoken cl100k_base encoding (GPT-4 compatible)
//! - **Multiple Encoding Support**: Supports different OpenAI encodings
//! - **Content-Aware Estimation**: Handles code content more accurately than character counting
//! - **Budget Management**: Token budget allocation and tracking
//!
//! ## Usage
//!
//! ```rust
//! use scribe_core::tokenization::{TokenCounter, TokenizerConfig};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = TokenizerConfig::default();
//! let counter = TokenCounter::new(config)?;
//!
//! let content = "fn main() { println!(\"Hello, world!\"); }";
//! let token_count = counter.count_tokens(content)?;
//! println!("Token count: {}", token_count);
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;
use tiktoken_rs::{get_bpe_from_model, CoreBPE};
use serde::{Deserialize, Serialize};
use once_cell::sync::Lazy;
use crate::Result;

/// Configuration for the tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// The encoding model to use (e.g., "gpt-4", "gpt-3.5-turbo")
    pub encoding_model: String,
    /// Whether to cache tokenizer instances
    pub enable_caching: bool,
    /// Token budget for content selection
    pub token_budget: Option<usize>,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            encoding_model: "gpt-4".to_string(),
            enable_caching: true,
            token_budget: Some(128000), // Default to GPT-4 context window
        }
    }
}

/// Shared global instance of the default TokenCounter (GPT-4)
/// This avoids expensive re-initialization on every token counting call
static GLOBAL_TOKEN_COUNTER: Lazy<TokenCounter> = Lazy::new(|| {
    TokenCounter::new(TokenizerConfig::default())
        .expect("Failed to initialize global token counter")
});

/// Main tokenizer interface for accurate token counting
pub struct TokenCounter {
    config: TokenizerConfig,
    bpe: Arc<CoreBPE>,
}

impl std::fmt::Debug for TokenCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenCounter")
            .field("config", &self.config)
            .field("bpe", &"<CoreBPE>")
            .finish()
    }
}

impl TokenCounter {
    /// Create a new token counter with the specified configuration
    pub fn new(config: TokenizerConfig) -> Result<Self> {
        let bpe = get_bpe_from_model(&config.encoding_model)
            .map_err(|e| crate::ScribeError::tokenization(format!("Failed to load tokenizer for model '{}': {}", config.encoding_model, e)))?;
        
        Ok(Self {
            config,
            bpe: Arc::new(bpe),
        })
    }
    
    /// Create a new token counter with default configuration (GPT-4)
    pub fn default() -> Result<Self> {
        Self::new(TokenizerConfig::default())
    }
    
    /// Get a reference to the shared global token counter instance
    /// This is highly optimized and avoids re-initialization costs
    pub fn global() -> &'static TokenCounter {
        &GLOBAL_TOKEN_COUNTER
    }
    
    /// Count tokens in the given text content
    pub fn count_tokens(&self, content: &str) -> Result<usize> {
        let tokens = self.bpe.encode_with_special_tokens(content);
        Ok(tokens.len())
    }
    
    /// Count tokens in multiple content strings and return the total
    pub fn count_tokens_batch(&self, contents: &[&str]) -> Result<usize> {
        let mut total = 0;
        for content in contents {
            total += self.count_tokens(content)?;
        }
        Ok(total)
    }
    
    /// Estimate tokens for a file based on its content and metadata
    pub fn estimate_file_tokens(&self, content: &str, file_path: &std::path::Path) -> Result<usize> {
        // Get base token count
        let base_tokens = self.count_tokens(content)?;
        
        // Apply language-specific multipliers based on file extension
        let multiplier = self.get_language_multiplier(file_path);
        
        Ok((base_tokens as f64 * multiplier).ceil() as usize)
    }
    
    /// Get language-specific token multiplier
    fn get_language_multiplier(&self, file_path: &std::path::Path) -> f64 {
        let extension = file_path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");
        
        match extension {
            // Languages with lots of boilerplate tend to have lower token density
            "java" | "csharp" | "cs" => 1.2,
            
            // Languages with compact syntax
            "py" | "python" => 0.9,
            "js" | "javascript" | "ts" | "typescript" => 0.95,
            "rs" | "rust" => 1.0,
            "go" => 0.95,
            
            // Configuration and data files
            "json" | "yaml" | "yml" | "toml" => 0.8,
            "xml" | "html" | "htm" => 1.1,
            
            // Documentation
            "md" | "markdown" | "txt" => 0.7,
            
            // Default for unknown types
            _ => 1.0,
        }
    }
    
    /// Check if content fits within the token budget
    pub fn fits_budget(&self, content: &str) -> Result<bool> {
        if let Some(budget) = self.config.token_budget {
            let token_count = self.count_tokens(content)?;
            Ok(token_count <= budget)
        } else {
            Ok(true) // No budget limit
        }
    }
    
    /// Calculate remaining budget after accounting for content
    pub fn remaining_budget(&self, used_tokens: usize) -> Option<usize> {
        self.config.token_budget.map(|budget| budget.saturating_sub(used_tokens))
    }
    
    /// Split content into chunks that fit within a token limit
    pub fn chunk_content(&self, content: &str, chunk_size: usize) -> Result<Vec<String>> {
        let tokens = self.bpe.encode_with_special_tokens(content);
        let mut chunks = Vec::new();
        
        for chunk_tokens in tokens.chunks(chunk_size) {
            let chunk_text = self.bpe.decode(chunk_tokens.to_vec())
                .map_err(|e| crate::ScribeError::tokenization(format!("Failed to decode token chunk: {}", e)))?;
            chunks.push(chunk_text);
        }
        
        Ok(chunks)
    }
    
    /// Get the current tokenizer configuration
    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }
    
    /// Update the token budget
    pub fn set_token_budget(&mut self, budget: Option<usize>) {
        self.config.token_budget = budget;
    }
}

/// Token budget tracker for selection algorithms
#[derive(Debug, Clone)]
pub struct TokenBudget {
    total_budget: usize,
    used_tokens: usize,
    reserved_tokens: usize,
}

impl TokenBudget {
    /// Create a new token budget tracker
    pub fn new(total_budget: usize) -> Self {
        Self {
            total_budget,
            used_tokens: 0,
            reserved_tokens: 0,
        }
    }
    
    /// Get the total budget
    pub fn total(&self) -> usize {
        self.total_budget
    }
    
    /// Get the number of tokens used
    pub fn used(&self) -> usize {
        self.used_tokens
    }
    
    /// Get the number of tokens reserved but not yet used
    pub fn reserved(&self) -> usize {
        self.reserved_tokens
    }
    
    /// Get the number of available tokens
    pub fn available(&self) -> usize {
        self.total_budget.saturating_sub(self.used_tokens + self.reserved_tokens)
    }
    
    /// Check if the budget can accommodate the specified number of tokens
    pub fn can_allocate(&self, tokens: usize) -> bool {
        self.available() >= tokens
    }
    
    /// Allocate tokens from the budget
    pub fn allocate(&mut self, tokens: usize) -> bool {
        if self.can_allocate(tokens) {
            self.used_tokens += tokens;
            true
        } else {
            false
        }
    }
    
    /// Reserve tokens without using them yet
    pub fn reserve(&mut self, tokens: usize) -> bool {
        if self.available() >= tokens {
            self.reserved_tokens += tokens;
            true
        } else {
            false
        }
    }
    
    /// Confirm reserved tokens as used
    pub fn confirm_reservation(&mut self, tokens: usize) {
        let to_confirm = tokens.min(self.reserved_tokens);
        self.reserved_tokens -= to_confirm;
        self.used_tokens += to_confirm;
    }
    
    /// Release reserved tokens back to available pool
    pub fn release_reservation(&mut self, tokens: usize) {
        self.reserved_tokens = self.reserved_tokens.saturating_sub(tokens);
    }
    
    /// Get utilization as a percentage
    pub fn utilization(&self) -> f64 {
        (self.used_tokens as f64 / self.total_budget as f64) * 100.0
    }
    
    /// Reset the budget tracker
    pub fn reset(&mut self) {
        self.used_tokens = 0;
        self.reserved_tokens = 0;
    }
}

/// Utilities for working with tokens and content
pub mod utils {
    use super::*;
    
    /// Estimate tokens using the legacy character-based method (for comparison)
    pub fn estimate_tokens_legacy(content: &str) -> usize {
        // Original method: ~4 characters per token for English text
        (content.chars().count() as f64 / 4.0).ceil() as usize
    }
    
    /// Compare tiktoken accuracy against legacy estimation
    pub fn compare_tokenization_accuracy(content: &str, counter: &TokenCounter) -> Result<TokenizationComparison> {
        let tiktoken_count = counter.count_tokens(content)?;
        let legacy_count = estimate_tokens_legacy(content);
        
        let accuracy_ratio = if legacy_count > 0 {
            tiktoken_count as f64 / legacy_count as f64
        } else {
            1.0
        };
        
        Ok(TokenizationComparison {
            tiktoken_count,
            legacy_count,
            accuracy_ratio,
            improvement: if accuracy_ratio < 1.0 { 
                Some((1.0 - accuracy_ratio) * 100.0)
            } else { 
                None 
            },
        })
    }
    
    /// Get recommended token budget based on model and content type
    pub fn recommend_token_budget(model: &str, content_type: ContentType) -> usize {
        let base_budget = match model {
            "gpt-4" | "gpt-4-turbo" => 128000,
            "gpt-4-32k" => 32000,
            "gpt-3.5-turbo" => 16000,
            "gpt-3.5-turbo-16k" => 16000,
            _ => 8000, // Conservative default
        };
        
        // Adjust based on content type
        match content_type {
            ContentType::Code => (base_budget as f64 * 0.8) as usize, // Leave room for analysis
            ContentType::Documentation => base_budget,
            ContentType::Mixed => (base_budget as f64 * 0.9) as usize,
        }
    }
}

/// Content type for budget recommendations
#[derive(Debug, Clone, Copy)]
pub enum ContentType {
    Code,
    Documentation,
    Mixed,
}

/// Comparison between tiktoken and legacy tokenization
#[derive(Debug, Clone)]
pub struct TokenizationComparison {
    pub tiktoken_count: usize,
    pub legacy_count: usize,
    pub accuracy_ratio: f64,
    pub improvement: Option<f64>, // Percentage improvement if tiktoken is more accurate
}

impl TokenizationComparison {
    /// Format the comparison as a human-readable string
    pub fn format(&self) -> String {
        match self.improvement {
            Some(improvement) => format!(
                "Tiktoken: {} tokens, Legacy: {} tokens, {:.1}% more accurate",
                self.tiktoken_count, self.legacy_count, improvement
            ),
            None => format!(
                "Tiktoken: {} tokens, Legacy: {} tokens, {:.2}x ratio",
                self.tiktoken_count, self.legacy_count, self.accuracy_ratio
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    
    #[test]
    fn test_token_counter_creation() {
        let config = TokenizerConfig::default();
        let counter = TokenCounter::new(config);
        assert!(counter.is_ok());
    }
    
    #[test]
    fn test_basic_token_counting() {
        let counter = TokenCounter::default().unwrap();
        
        let simple_text = "Hello, world!";
        let count = counter.count_tokens(simple_text).unwrap();
        assert!(count > 0);
        assert!(count < 10); // Should be a small number for this simple text
    }
    
    #[test]
    fn test_code_token_counting() {
        let counter = TokenCounter::default().unwrap();
        
        let rust_code = r#"
fn main() {
    println!("Hello, world!");
    let x = 42;
    if x > 0 {
        println!("Positive number: {}", x);
    }
}
"#;
        
        let count = counter.count_tokens(rust_code).unwrap();
        assert!(count > 20); // Should be more tokens for this code
        assert!(count < 100); // But not excessive
    }
    
    #[test]
    fn test_language_multipliers() {
        let counter = TokenCounter::default().unwrap();
        
        let content = "function test() { return 42; }";
        
        let js_tokens = counter.estimate_file_tokens(content, Path::new("test.js")).unwrap();
        let java_tokens = counter.estimate_file_tokens(content, Path::new("test.java")).unwrap();
        let py_tokens = counter.estimate_file_tokens(content, Path::new("test.py")).unwrap();
        
        // Java should have more tokens due to boilerplate multiplier
        assert!(java_tokens >= js_tokens);
        // Python should have fewer tokens due to compact syntax
        assert!(py_tokens <= js_tokens);
    }
    
    #[test]
    fn test_token_budget() {
        let mut budget = TokenBudget::new(1000);
        
        assert_eq!(budget.total(), 1000);
        assert_eq!(budget.used(), 0);
        assert_eq!(budget.available(), 1000);
        
        assert!(budget.allocate(300));
        assert_eq!(budget.used(), 300);
        assert_eq!(budget.available(), 700);
        
        assert!(budget.reserve(200));
        assert_eq!(budget.reserved(), 200);
        assert_eq!(budget.available(), 500);
        
        budget.confirm_reservation(150);
        assert_eq!(budget.used(), 450);
        assert_eq!(budget.reserved(), 50);
        assert_eq!(budget.available(), 500);
    }
    
    #[test]
    fn test_content_chunking() {
        let counter = TokenCounter::default().unwrap();
        
        let long_content = "word ".repeat(1000); // 1000 words
        let chunks = counter.chunk_content(&long_content, 100).unwrap();
        
        assert!(chunks.len() > 1); // Should be split into multiple chunks
        
        // Verify each chunk is roughly the right size
        for chunk in &chunks {
            let chunk_tokens = counter.count_tokens(chunk).unwrap();
            assert!(chunk_tokens <= 120); // Allow some margin due to token boundaries
        }
    }
    
    #[test]
    fn test_tokenization_comparison() {
        let counter = TokenCounter::default().unwrap();
        
        let code_content = r#"
use std::collections::HashMap;

fn process_data(input: &str) -> Result<HashMap<String, i32>, Box<dyn std::error::Error>> {
    let mut result = HashMap::new();
    for line in input.lines() {
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() == 2 {
            result.insert(parts[0].to_string(), parts[1].parse()?);
        }
    }
    Ok(result)
}
"#;
        
        let comparison = utils::compare_tokenization_accuracy(code_content, &counter).unwrap();
        
        assert!(comparison.tiktoken_count > 0);
        assert!(comparison.legacy_count > 0);
        assert!(comparison.accuracy_ratio > 0.0);
        
        let formatted = comparison.format();
        assert!(formatted.contains("Tiktoken"));
        assert!(formatted.contains("Legacy"));
    }
    
    #[test]
    fn test_budget_recommendations() {
        let code_budget = utils::recommend_token_budget("gpt-4", ContentType::Code);
        let doc_budget = utils::recommend_token_budget("gpt-4", ContentType::Documentation);
        let mixed_budget = utils::recommend_token_budget("gpt-4", ContentType::Mixed);
        
        assert!(code_budget < doc_budget); // Code should have smaller budget to leave room for analysis
        assert!(mixed_budget > code_budget);
        assert!(mixed_budget < doc_budget);
    }
}
# Scribe Core

Core types, utilities, and foundational components for the Scribe code analysis library.

## Overview

`scribe-core` provides the fundamental data structures and traits used across all other Scribe crates. It includes comprehensive error handling, file analysis types, scoring components, configuration management, and extensibility traits.

## Features

- **Comprehensive Error Handling**: Rich error types with proper context and error chaining
- **File Analysis**: File metadata, language detection, and classification systems
- **Scoring System**: Heuristic scoring components and configurable weights
- **Configuration**: Flexible configuration with validation and serialization support
- **Extensibility**: Traits for custom analyzers, scorers, and formatters
- **Utilities**: Common functions for path manipulation, string processing, and more

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
scribe-core = "0.1.0"
```

Basic usage:

```rust
use scribe_core::{Config, Language, ScoreComponents, HeuristicWeights, Result};

// Create a default configuration
let config = Config::default();

// Detect programming language from file extension
let language = Language::from_extension("rs");
assert_eq!(language, Language::Rust);

// Work with scoring components
let mut scores = ScoreComponents::zero();
scores.doc_score = 0.8;
scores.import_score = 0.6;

let weights = HeuristicWeights::default();
scores.compute_final_score(&weights);

println!("Final score: {}", scores.final_score);
```

## Key Types

### Error Handling
- `ScribeError`: Comprehensive error type for all Scribe operations
- `Result<T>`: Type alias for `std::result::Result<T, ScribeError>`

### File Analysis
- `FileInfo`: Complete file metadata and analysis results
- `Language`: Programming language detection and classification
- `FileType`: File type classification (source, test, documentation, etc.)
- `RenderDecision`: Include/exclude decisions for files

### Scoring System
- `ScoreComponents`: Individual heuristic score components
- `HeuristicWeights`: Configurable weights for scoring
- `RepositoryInfo`: Repository-level statistics and metadata

### Configuration
- `Config`: Main configuration structure
- `FeatureFlags`: Enable/disable experimental features
- Various sub-configurations for filtering, analysis, performance, etc.

### Traits for Extensibility
- `FileAnalyzer`: Custom file analysis implementations
- `HeuristicScorer`: Custom scoring algorithms
- `OutputFormatter`: Custom output formats
- `CacheStorage`: Custom caching implementations

## Architecture

The crate is organized into several modules:

- `error`: Comprehensive error types and handling
- `file`: File metadata and analysis types
- `types`: Core data structures and scoring components
- `config`: Configuration management
- `traits`: Extensibility traits and abstractions
- `utils`: Common utility functions

## Memory Safety and Performance

Scribe Core is built with Rust's ownership principles in mind:

- **Zero-cost abstractions**: Efficient implementations without runtime overhead
- **Memory safety**: Compile-time guarantees prevent common bugs
- **Thread safety**: Safe concurrent operations where appropriate
- **Resource efficiency**: Minimal memory allocations in hot paths

## Testing

Run the test suite:

```bash
cargo test
```

Run benchmarks:

```bash
cargo bench
```

## Contributing

This crate serves as the foundation for the entire Scribe ecosystem. When adding new functionality:

1. Ensure comprehensive error handling using `ScribeError`
2. Add proper documentation and examples
3. Include unit tests for all new functionality
4. Consider thread safety and performance implications
5. Follow existing naming conventions and patterns

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
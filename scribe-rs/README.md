# Scribe - Advanced Code Analysis Library

[![Crates.io](https://img.shields.io/crates/v/scribe.svg)](https://crates.io/crates/scribe)
[![Documentation](https://docs.rs/scribe/badge.svg)](https://docs.rs/scribe)
[![License](https://img.shields.io/crates/l/scribe.svg)](https://github.com/sibyllinesoft/scribe#license)
[![Build Status](https://github.com/sibyllinesoft/scribe/workflows/CI/badge.svg)](https://github.com/sibyllinesoft/scribe/actions)

Scribe is a comprehensive Rust library for code analysis, repository exploration, and intelligent file processing. It provides powerful tools for understanding codebases through heuristic scoring, graph analysis, and AI-powered insights.

## 🚀 Features

- **🔍 Intelligent File Analysis**: Multi-dimensional heuristic scoring system for identifying important files
- **📊 Dependency Graph Analysis**: PageRank centrality computation for understanding code relationships
- **⚡ High-Performance Scanning**: Parallel file system traversal with git integration
- **🎯 Advanced Pattern Matching**: Flexible glob and gitignore pattern support with preset configurations
- **🧠 Smart Code Selection**: Context-aware code bundling and relevance scoring
- **🛠️ Extensible Architecture**: Plugin system for custom analyzers and scorers
- **⚙️ Modular Design**: Use only the features you need with optional components

## 📦 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
scribe = "0.1.0"
```

### Feature Flags

Scribe uses feature flags to allow selective compilation:

```toml
# Full installation (default)
scribe = "0.1.0"

# Minimal installation
scribe = { version = "0.1.0", default-features = false, features = ["core"] }

# Fast file operations only
scribe = { version = "0.1.0", default-features = false, features = ["fast"] }

# Analysis without graph features
scribe = { version = "0.1.0", default-features = false, features = ["core", "analysis", "scanner"] }
```

#### Available Features

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `default` | All features enabled | `core`, `analysis`, `graph`, `scanner`, `patterns`, `selection` |
| `core` | Essential types and utilities | None |
| `analysis` | Heuristic scoring and metrics | `core` |
| `graph` | PageRank centrality analysis | `core`, `analysis` |
| `scanner` | File system scanning | `core` |
| `patterns` | Pattern matching (glob, gitignore) | `core` |
| `selection` | Code selection and bundling | `core`, `analysis`, `graph` |

#### Feature Groups

| Group | Features | Use Case |
|-------|----------|----------|
| `minimal` | `core` | Basic types and utilities only |
| `fast` | `core`, `scanner`, `patterns` | Quick file operations |
| `comprehensive` | All features | Complete analysis capabilities |

## 🏃 Quick Start

### Basic Repository Analysis

```rust
use scribe::prelude::*;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    // Analyze a repository with default settings
    let config = Config::default();
    let analysis = analyze_repository(".", &config).await?;
    
    // Get the most important files
    println!("Top 10 most important files:");
    for (file, score) in analysis.top_files(10) {
        println!("  {}: {:.3}", file, score);
    }
    
    // Display summary
    println!("\n{}", analysis.summary());
    
    Ok(())
}
```

### Selective Feature Usage

```rust
// Using only core and scanner features
use scribe::core::{Config, Result};
use scribe::scanner::{Scanner, ScanOptions};

#[tokio::main]
async fn main() -> Result<()> {
    let scanner = Scanner::new();
    let options = ScanOptions::default()
        .with_git_integration(true)
        .with_parallel_processing(true);
    
    let files = scanner.scan(".", options).await?;
    println!("Found {} files", files.len());
    
    Ok(())
}
```

### Pattern Matching

```rust
use scribe::patterns::presets;

#[tokio::main]
async fn main() -> scribe::Result<()> {
    // Use preset patterns for common file types
    let mut source_matcher = presets::source_code()?;
    let mut doc_matcher = presets::documentation()?;
    
    if source_matcher.should_process("src/main.rs")? {
        println!("Found source file!");
    }
    
    if doc_matcher.should_process("README.md")? {
        println!("Found documentation!");
    }
    
    Ok(())
}
```

### Graph Analysis

```rust
use scribe::graph::PageRankAnalysis;

#[tokio::main]
async fn main() -> scribe::Result<()> {
    let analysis = PageRankAnalysis::for_code_analysis()?;
    
    // Compute centrality for scan results
    // let centrality_results = analysis.compute_centrality(&scan_results)?;
    // let top_files = centrality_results.top_files_by_centrality(10);
    
    Ok(())
}
```

### CLI Covering Sets

Scribe’s CLI can compute minimal covering sets:

- `--covering-set <name>`: target a function/class/module by name.
- `--covering-set-diff`: build a covering set for the current `git diff` (uses the dependency graph to include touched files plus related dependents/dependencies).
- `--diff-against <ref>`: diff against a specific ref (defaults to `HEAD`).
- Shared filters: `--include-dependents`, `--max-depth`, `--max-files`.
- Output helper: add `--line-numbers` to prefix every line in the bundled files, making it easy for review agents to comment by line number.

Example:

```bash
cargo run --bin scribe -- --covering-set-diff --include-dependents --max-depth 2
```

## 🏗️ Architecture

Scribe is built with a modular architecture where each crate provides specific functionality:

```
┌─────────────────────────────────────────────────────────────┐
│                        scribe                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │ scribe-core │ │scribe-scanner│ │    scribe-patterns     │ │
│  │   (types,   │ │(file system  │ │  (glob, gitignore,     │ │
│  │ traits,     │ │ traversal,   │ │   pattern matching)    │ │
│  │ utilities)  │ │ git support) │ │                        │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │scribe-analysis│ │scribe-graph │ │   scribe-selection     │ │
│  │ (heuristic  │ │  (PageRank  │ │ (intelligent bundling, │ │
│  │  scoring,   │ │ centrality, │ │  context extraction,   │ │
│  │ code metrics)│ │ dependency  │ │   relevance scoring)   │ │
│  │             │ │  analysis)  │ │                        │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Component Overview

- **`scribe-core`**: Foundation types, traits, configuration, and utilities
- **`scribe-scanner`**: High-performance file system traversal with git integration
- **`scribe-patterns`**: Flexible pattern matching with glob and gitignore support
- **`scribe-analysis`**: Heuristic scoring algorithms and code metrics
- **`scribe-graph`**: PageRank centrality and dependency graph analysis
- **`scribe-selection`**: Intelligent code selection and context extraction

## 📖 Examples

The repository includes several examples demonstrating different usage patterns:

### Run Examples

```bash
# Full analysis example
cargo run --example basic_usage -- /path/to/repository

# Minimal features example  
cargo run --example selective_features --no-default-features --features="core,scanner" -- /path/to/directory
```

### Available Examples

- **`basic_usage.rs`**: Complete repository analysis with all features
- **`selective_features.rs`**: Minimal usage with core and scanner only

## 🔧 Performance

Scribe is designed for high performance:

- **Memory Efficient**: Streaming file processing with configurable memory limits
- **Parallel Processing**: Multi-threaded scanning and analysis using Rayon
- **Git Integration**: Fast file discovery using `git ls-files` when available
- **Optimized Algorithms**: Research-grade PageRank implementation with convergence detection

### Benchmarks

Run benchmarks to see performance characteristics:

```bash
cargo bench
```

Performance characteristics on typical repositories:

- **Small repos (< 1k files)**: ~10-50ms analysis time
- **Medium repos (1k-10k files)**: ~100ms-1s analysis time  
- **Large repos (> 10k files)**: ~1-10s analysis time
- **Memory usage**: ~2MB per 1000 files for basic analysis

## 🛠️ Development

### Building

```bash
# Build all features
cargo build

# Build with specific features
cargo build --no-default-features --features="core,scanner"

# Build for release
cargo build --release
```

### Testing

```bash
# Run all tests
cargo test

# Test specific features
cargo test --no-default-features --features="core,analysis"

# Run tests with output
cargo test -- --nocapture
```

### Documentation

```bash
# Generate documentation
cargo doc --open

# Generate documentation for all features
cargo doc --all-features --open
```

## 🔗 Related Projects

- **[scribe-cli]**: Command-line interface for Scribe
- **[scribe-vscode]**: Visual Studio Code extension
- **[scribe-jupyter]**: Jupyter notebook integration

## 📄 License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📞 Support

- 📖 **Documentation**: [docs.rs/scribe](https://docs.rs/scribe)
- 🐛 **Issues**: [GitHub Issues](https://github.com/sibyllinesoft/scribe/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/sibyllinesoft/scribe/discussions)

## 🙏 Acknowledgments

- Built with [Rust](https://rust-lang.org/) 🦀
- Uses [tree-sitter](https://tree-sitter.github.io/) for parsing
- Inspired by research in code analysis and repository mining
- Community feedback and contributions

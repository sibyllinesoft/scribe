# Crate Reference

Detailed documentation for each crate in the Scribe workspace.

## scribe-core

**Purpose**: Shared domain types and configuration logic.

### Key Types

- `FileInfo`: Metadata for a scanned file
- `Language`: Supported programming languages
- `FileType`: Classification (Source, Test, Config, etc.)
- `RenderDecision`: Whether/how to include a file
- `Config`: Global configuration

### Usage

```rust
use scribe_core::{FileInfo, Language, Config};

let config = Config::default();
let file = FileInfo::new(path, &config)?;
```

---

## scribe-patterns

**Purpose**: Glob patterns and gitignore handling.

### Features

- Fast glob matching with caching
- Full gitignore spec support
- Composable pattern sets

### Usage

```rust
use scribe_patterns::{GlobMatcher, GitignoreBuilder};

let matcher = GlobMatcher::new(&["**/*.rs", "!**/test_*.rs"])?;
let gitignore = GitignoreBuilder::new(repo_root).build()?;
```

---

## scribe-scanner

**Purpose**: Repository traversal, filtering, and language detection.

### Features

- Parallel directory walking (via `ignore` crate)
- Binary file detection
- Language detection by extension and content
- Size and depth limits

### Usage

```rust
use scribe_scanner::{Scanner, ScanConfig};

let config = ScanConfig::default();
let scanner = Scanner::new(repo_path, config)?;
let files = scanner.scan()?;
```

---

## scribe-analysis

**Purpose**: File content analysis and tree-sitter AST parsing.

### Features

- Entity extraction (functions, classes, types)
- Import/export detection
- Complexity metrics
- Semantic chunking for demotion

### Supported Languages

| Language | Parser | Entity Extraction |
|----------|--------|-------------------|
| Rust | tree-sitter-rust | Full |
| Python | tree-sitter-python | Full |
| TypeScript | tree-sitter-typescript | Full |
| JavaScript | tree-sitter-javascript | Full |
| Go | tree-sitter-go | Full |
| Java | tree-sitter-java | Partial |

### Usage

```rust
use scribe_analysis::{Analyzer, AnalysisConfig};

let analyzer = Analyzer::new(AnalysisConfig::default());
let result = analyzer.analyze_file(&file_info)?;

for entity in result.entities {
    println!("{}: {}", entity.kind, entity.name);
}
```

---

## scribe-graph

**Purpose**: Import graph construction and graph algorithms.

### Features

- Dependency graph from imports
- PageRank centrality
- Betweenness centrality
- Strongly connected components (Kosaraju)
- Transitive closure computation

### Usage

```rust
use scribe_graph::{DependencyGraph, CentralityCalculator};

let graph = DependencyGraph::build(&files)?;
let centrality = CentralityCalculator::pagerank(&graph, 0.85, 100)?;

for (file, score) in centrality.top(10) {
    println!("{}: {:.4}", file, score);
}
```

---

## scribe-selection

**Purpose**: Selection heuristics and covering set computation.

### Algorithms

- **Simple**: Size and path depth scoring
- **Complex**: Multi-dimensional heuristics
- **Heuristic**: Balanced approach with centrality
- **CoveringSet**: Transitive dependency analysis

### Usage

```rust
use scribe_selection::{Selector, SelectionConfig, CoveringSetTarget};

// Repository-wide selection
let selector = Selector::new(SelectionConfig::default());
let selected = selector.select(&files, token_budget)?;

// Covering set for specific target
let target = CoveringSetTarget::entity("src/auth.rs", "authenticate_user");
let covering = selector.compute_covering_set(&files, &target)?;
```

---

## scribe-scaling

**Purpose**: Token budgeting, progressive demotion, and context positioning.

### Features

- Token estimation (tiktoken-compatible)
- Progressive demotion: Full → Chunks → Signatures
- Context positioning (HEAD/MIDDLE/TAIL)
- Budget utilization optimization

### Usage

```rust
use scribe_scaling::{ScalingSelector, ScalingConfig, ContextPositioningConfig};

let mut config = ScalingConfig::default();
config.token_budget = 100_000;
config.positioning = ContextPositioningConfig::default();

let selector = ScalingSelector::new(config);
let result = selector.select_and_scale(&files).await?;

println!("Used {} tokens ({:.1}% of budget)",
    result.total_tokens,
    result.budget_utilization * 100.0);
```

---

## scribe-webservice

**Purpose**: Axum-based HTTP API for the bundle editor.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Interactive bundle editor |
| GET | `/api/bundle` | Get current bundle state |
| POST | `/api/bundle` | Update bundle configuration |
| GET | `/api/files` | List available files |
| GET | `/api/analyze` | Trigger re-analysis |

### Usage

```rust
use scribe_webservice::{WebService, WebConfig};

let config = WebConfig {
    port: 8080,
    repo_path: PathBuf::from("."),
    ..Default::default()
};

let service = WebService::new(config)?;
service.run().await?;
```

---

## scribe (main crate)

**Purpose**: CLI entry point and library facade.

### Library Usage

```rust
use scribe::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let config = Config::default();
    let analysis = analyze_repository(".", &config).await?;

    // Top files by centrality
    for (file, score) in analysis.top_files(10) {
        println!("{}: {:.3}", file, score);
    }

    // Covering set
    let target = CoveringSetTarget::entity("src/main.rs", "main");
    let covering = analysis.covering_set(&target)?;

    for file in covering.files {
        println!("{} ({})", file.path.display(), file.reason);
    }

    Ok(())
}
```

### Feature Flags

```toml
[dependencies]
# Full installation (default)
scribe = "0.5"

# Minimal - core types only
scribe = { version = "0.5", default-features = false, features = ["core"] }

# Analysis without web features
scribe = { version = "0.5", default-features = false, features = ["core", "analysis", "selection"] }
```

---

## Cargo Features

| Feature | Crates Enabled | Use Case |
|---------|---------------|----------|
| `default` | All | Full CLI functionality |
| `core` | scribe-core | Type definitions only |
| `analysis` | +scribe-analysis | AST parsing |
| `graph` | +scribe-graph | Dependency graphs |
| `selection` | +scribe-selection | Selection algorithms |
| `scaling` | +scribe-scaling | Token management |
| `web` | +scribe-webservice | HTTP API |

# scribe-selection

Intelligent file selection and context extraction for Scribe repository bundles.

## Overview

`scribe-selection` implements sophisticated algorithms for choosing which files to include in a repository bundle. Rather than naively including everything or requiring manual selection, it uses multi-dimensional scoring, graph centrality, and heuristic analysis to automatically identify the most important code for LLM understanding.

## Key Features

### Multi-Dimensional Scoring
- **Documentation score**: Rewards READMEs, design docs, API documentation
- **PageRank centrality**: Files that are heavily imported get higher scores
- **Test linkage**: Identifies test files and their corresponding source files
- **Git churn**: Integrates change frequency and recency signals
- **Path depth**: Closer-to-root files prioritized (entry points, configs)
- **Template detection**: Demotes boilerplate and auto-generated code
- **Entrypoint detection**: Identifies `main.rs`, `__init__.py`, `index.js`
- **Example code scoring**: Detects and prioritizes example/demo code

### Selection Algorithms

#### Simple Router (Default)
Transparent rule-based decision tree for file selection:
- Priority-based routing (mandatory → high priority → budget-based)
- Deterministic and explainable decisions
- Fast: O(n log n) for sorting, O(n) for selection
- Replaced complex bandit algorithm with clearer logic

#### Covering Set Selection
Surgical selection targeting specific entities:
- AST-based search for functions, classes, modules
- Transitive dependency closure computation
- Configurable depth limits and importance thresholds
- Minimal file sets for understanding or impact analysis

### Progressive Demotion
When approaching token budgets, intelligently reduce content:
- **FULL**: Complete file content (default)
- **CHUNK**: AST-based semantic sections (important functions/classes preserved)
- **SIGNATURE**: Type signatures and interfaces only (minimal information)

Achieves 3-10x compression while preserving critical context.

### Configurable Weights
Two preset configurations:
- **V1**: Balanced across all dimensions
- **V2**: Higher weight on centrality and documentation

Custom weight tuning for specific use cases.

## Architecture

```
FileMetadata + Scores → Selection Algorithm → Budget Enforcement → Demotion Engine → Final Selection
      ↓            ↓               ↓                  ↓                  ↓                ↓
  Heuristics   PageRank      Simple Router      Token Check         AST Chunking    Selected Files
   Scoring     Analysis      Decision Tree      Hard Limits        Signature Extract   with Metadata
```

### Core Components

#### `FileScorer`
Computes multi-dimensional importance scores:
```rust
score = w_doc*doc + w_readme*readme + w_imp*imp_deg + w_path*path_depth^-1 +
        w_test*test_link + w_churn*churn + w_centrality*centrality +
        w_entrypoint*entrypoint + w_examples*examples + priority_boost
```

Each dimension is normalized to [0, 1] and combined with configurable weights.

#### `SimpleRouter`
Rule-based selection algorithm:
1. **Mandatory files**: README, LICENSE, config files (always included)
2. **High priority**: Files with score > threshold
3. **Budget-based**: Fill remaining budget with highest-scoring files
4. **Test handling**: Optional inclusion based on configuration

#### `CoveringSetSelector`
Targets specific code entities:
- Entity search using tree-sitter AST parsing
- Graph traversal for dependencies/dependents
- Importance filtering to exclude noise
- Reason tracking for transparency

#### `DemotionEngine`
Progressive content reduction:
- **Full → Chunk**: Extract high-importance functions/classes via AST
- **Chunk → Signature**: Keep only type definitions and interfaces
- **Quality scoring**: Tracks information preservation ratio

## Usage

### Basic File Selection

```rust
use scribe_selection::{Selector, SelectionConfig};

let config = SelectionConfig {
    algorithm: Algorithm::SimpleRouter,
    token_budget: 100_000,
    max_files: Some(200),
    exclude_tests: true,
    ..Default::default()
};

let selector = Selector::new(config);
let result = selector.select(files).await?;

println!("Selected {} files using {} tokens",
    result.selected.len(),
    result.total_tokens
);
```

### Custom Scoring Weights

```rust
use scribe_selection::{ScoringWeights, SelectionConfig};

let weights = ScoringWeights {
    documentation: 0.3,
    centrality: 0.4,      // Emphasize graph importance
    test_linkage: 0.1,
    churn: 0.1,
    path_depth: 0.05,
    entrypoint: 0.05,
};

let config = SelectionConfig {
    scoring_weights: weights,
    ..Default::default()
};

let selector = Selector::new(config);
```

### Covering Set for Entity

```rust
use scribe_selection::{CoveringSetConfig, EntityType};

let config = CoveringSetConfig {
    entity_name: "authenticate_user".to_string(),
    entity_type: EntityType::Function,
    max_files: 20,
    max_depth: Some(3),
    include_dependents: false,  // For understanding mode
    importance_threshold: 0.01,
};

let result = selector.select_covering_set(files, config).await?;

for (file, reason) in result.selected {
    println!("{}: {:?}", file.path.display(), reason);
}
// Output:
//   src/auth.rs: Target (contains function)
//   src/db.rs: DirectDependency (imported by auth.rs)
//   src/config.rs: TransitiveDependency (imported by db.rs, depth 2)
```

### Progressive Demotion

```rust
use scribe_selection::{DemotionLevel, DemotionConfig};

let mut config = SelectionConfig::default();
config.demotion_enabled = true;
config.demotion_threshold = 0.9; // Start demoting at 90% of budget

let selector = Selector::new(config);
let result = selector.select_with_budget(files, 50_000).await?;

// Check demotion results
for file in &result.selected {
    match file.demotion_level {
        DemotionLevel::Full => println!("{}: full content", file.path.display()),
        DemotionLevel::Chunk => println!("{}: chunked to key sections", file.path.display()),
        DemotionLevel::Signature => println!("{}: signatures only", file.path.display()),
    }
}

println!("Compression ratio: {:.2}x", result.compression_ratio);
println!("Quality score: {:.2}%", result.quality_score * 100.0);
```

### Impact Analysis Mode

```rust
use scribe_selection::{CoveringSetConfig, EntityType};

let config = CoveringSetConfig {
    entity_name: "User".to_string(),
    entity_type: EntityType::Class,
    include_dependents: true,  // Find what depends on this class
    max_depth: Some(2),
    ..Default::default()
};

let result = selector.select_covering_set(files, config).await?;

println!("Changing User class affects {} files:", result.selected.len());
for (file, reason) in result.selected {
    if matches!(reason, InclusionReason::Dependent(_)) {
        println!("  {} will be impacted", file.path.display());
    }
}
```

## Scoring Dimensions

### Documentation Score (0-1)
- **README files**: 1.0 (highest priority)
- **Design docs** (`DESIGN.md`, `ARCHITECTURE.md`): 0.9
- **API documentation**: 0.8
- **Code comments**: Proportional to comment density
- **Docstrings**: Bonus for well-documented code

### PageRank Centrality (0-1)
- Normalized PageRank score from dependency graph
- Files imported by many others get high scores
- Core utilities, config files typically score high
- Isolated files score low

### Test Linkage (0-1)
- Test files get score based on source file importance
- Source files with tests get bonus score
- Helps include relevant context for tested code

### Git Churn (0-1)
- **Change frequency**: More commits = higher activity
- **Recency**: Recent changes weighted higher
- **Combined score**: `churn = frequency * recency`

### Path Depth (0-1)
- Inverse of directory depth: `1 / (depth + 1)`
- Root-level files score highest
- Deeply nested utility files score lower
- Encourages including entry points and configs

### Template Detection (penalty)
- Auto-generated code detection
- Boilerplate pattern matching
- License headers, copyright notices
- Generated API clients, scaffolds
- **Penalty**: Reduces score by 50-80%

## Performance

### Targets
- **Selection time**: O(n log n) for sorting, O(n) for selection
- **Small repos (≤1k files)**: <100ms selection time
- **Medium repos (1k-10k)**: <500ms selection time
- **Large repos (10k-100k)**: <2s selection time

### Optimizations
- **Lazy scoring**: Only compute scores for files that pass initial filters
- **Parallel scoring**: Multi-threaded score computation using Rayon
- **Incremental demotion**: Progressive content reduction as budget fills
- **Score caching**: Cache expensive computations (centrality, churn)

## Configuration

### `SelectionConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `algorithm` | `Algorithm` | `SimpleRouter` | Selection algorithm to use |
| `token_budget` | `usize` | `100_000` | Maximum tokens in bundle |
| `max_files` | `Option<usize>` | `None` | Maximum number of files |
| `exclude_tests` | `bool` | `false` | Exclude test files |
| `scoring_weights` | `ScoringWeights` | `V2` | Weight configuration |
| `demotion_enabled` | `bool` | `true` | Enable progressive demotion |
| `demotion_threshold` | `f64` | `0.85` | Start demotion at % of budget |

### `ScoringWeights`

| Field | Type | V1 | V2 | Description |
|-------|------|----|----|-------------|
| `documentation` | `f64` | 0.2 | 0.25 | Documentation scoring weight |
| `centrality` | `f64` | 0.2 | 0.30 | PageRank centrality weight |
| `test_linkage` | `f64` | 0.15 | 0.10 | Test-source relationship weight |
| `churn` | `f64` | 0.15 | 0.10 | Git activity weight |
| `path_depth` | `f64` | 0.15 | 0.10 | Directory depth weight |
| `entrypoint` | `f64` | 0.10 | 0.10 | Entry point detection weight |
| `examples` | `f64` | 0.05 | 0.05 | Example code weight |

## Integration

`scribe-selection` is the core decision-making component used by:

- **scribe-scaling**: Applies selection within budget constraints
- **scribe-webservice**: Powers interactive file selection UI
- **CLI**: Implements `--algorithm`, `--token-budget`, `--max-files` flags
- **scribe-graph**: Uses centrality scores from graph analysis

## See Also

- `scribe-graph`: Provides PageRank centrality scores
- `scribe-scaling`: Token budgeting and performance optimization
- `scribe-analysis`: AST parsing for demotion and chunking
- `../../WHY_SCRIBE.md`: Context on intelligent selection philosophy

# Architecture

Scribe is built as a Rust workspace with specialized crates for each concern.

## High-Level Overview

```
scribe-rs/
├── scribe-core          # Shared domain types and configuration
├── scribe-scanner       # Repository traversal and file detection
├── scribe-analysis      # Content analysis and AST parsing
├── scribe-graph         # Import graph and graph algorithms
├── scribe-selection     # Selection strategies and covering sets
├── scribe-scaling       # Token budgeting and context positioning
├── scribe-webservice    # Axum-based API for bundle editor
└── scribe (lib+bin)     # CLI and library entry point
```

## Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Scanner   │ ──▶ │  Analysis   │ ──▶ │    Graph    │
│             │     │             │     │             │
│ Walk files  │     │ Parse AST   │     │ Build deps  │
│ Apply rules │     │ Extract     │     │ PageRank    │
│ Detect lang │     │ entities    │     │ Centrality  │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Output    │ ◀── │   Scaling   │ ◀── │  Selection  │
│             │     │             │     │             │
│ Markdown    │     │ Token mgmt  │     │ Algorithms  │
│ HTML/XML    │     │ Demotion    │     │ Covering    │
│ JSON        │     │ Positioning │     │ sets        │
└─────────────┘     └─────────────┘     └─────────────┘
```

### 1. Scanning (`scribe-scanner`)

Walks the repository applying:

- Gitignore rules
- Custom include/exclude patterns
- File size limits
- Binary file detection
- Language detection via extension and content

### 2. Analysis (`scribe-analysis`)

Uses tree-sitter for AST parsing:

- Extract functions, classes, types, constants
- Identify imports and exports
- Compute complexity metrics
- Generate semantic chunks for demotion

### 3. Graph (`scribe-graph`)

Builds the dependency graph:

- Import/export relationships
- Module hierarchy
- PageRank centrality computation
- Strongly connected component detection

### 4. Selection (`scribe-selection`)

Implements selection strategies:

- **Simple**: Basic heuristics (size, path depth)
- **Complex**: Multi-dimensional scoring
- **Covering Set**: Transitive dependency computation
- **Heuristic**: Balanced approach

### 5. Scaling (`scribe-scaling`)

Token budget management:

- Progressive demotion (full → chunks → signatures)
- Context positioning (HEAD/MIDDLE/TAIL)
- Quality score tracking
- Budget utilization optimization

### 6. Output (`scribe`)

Final rendering:

- Multiple formats (Markdown, HTML, XML, JSON)
- Interactive editor (HTML + embedded JS)
- Inclusion reason annotations
- Metadata and statistics

## Crate Dependencies

```
scribe-core (base types)
    │
    ├── scribe-patterns (glob, gitignore)
    │
    ├── scribe-scanner (uses core, patterns)
    │
    ├── scribe-analysis (uses core)
    │       │
    │       └── scribe-graph (uses core, analysis)
    │               │
    │               └── scribe-selection (uses core, graph)
    │                       │
    │                       └── scribe-scaling (uses core, selection)
    │
    └── scribe-webservice (uses all above)
            │
            └── scribe (CLI, uses all)
```

## Key Types

### `FileInfo` (scribe-core)

```rust
pub struct FileInfo {
    pub path: PathBuf,
    pub relative_path: String,
    pub size: u64,
    pub language: Language,
    pub file_type: FileType,
    pub token_estimate: Option<usize>,
    pub centrality_score: Option<f64>,
    pub content: Option<String>,
    // ...
}
```

### `CoveringSetResult` (scribe-selection)

```rust
pub struct CoveringSetResult {
    pub target: CoveringSetTarget,
    pub files: Vec<CoveringSetFile>,
    pub statistics: CoveringSetStats,
}

pub struct CoveringSetFile {
    pub path: PathBuf,
    pub distance: usize,
    pub reason: InclusionReason,
    pub entities: Vec<Entity>,
    pub content: String,
}
```

### `ScalingResult` (scribe-scaling)

```rust
pub struct ScalingResult {
    pub files: Vec<ScaledFile>,
    pub total_tokens: usize,
    pub budget_utilization: f64,
    pub positioning: Option<ContextPositioning>,
}
```

## Performance Characteristics

| Operation | Small Repo | Medium Repo | Large Repo |
|-----------|------------|-------------|------------|
| Scan | < 100ms | ~500ms | ~2s |
| AST Parse | ~200ms | ~1s | ~5s |
| Graph Build | ~50ms | ~200ms | ~1s |
| PageRank | ~10ms | ~50ms | ~100ms |
| Selection | ~10ms | ~50ms | ~200ms |
| **Total** | **< 500ms** | **~2s** | **~10s** |

Memory scales roughly linearly:

- ~50MB for small repos
- ~200MB for medium repos
- ~500MB-2GB for large repos

## Extending Scribe

### Adding a Selection Algorithm

1. Add variant to `SelectionAlgorithm` in `scribe-selection`
2. Implement `SelectionStrategy` trait
3. Wire up in CLI argument parsing

### Adding an Output Format

1. Add variant to `ReportFormat` in `scribe-core`
2. Implement renderer in `scribe/src/render/`
3. Add CLI option

### Adding Language Support

1. Add tree-sitter grammar dependency
2. Implement `LanguageParser` in `scribe-analysis`
3. Add import resolution in `scribe-graph`

## See Also

- [Crate Reference](crate-reference.md) - Detailed crate documentation
- [Contributing](contributing.md) - How to contribute

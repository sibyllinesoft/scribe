# scribe-graph

Graph-based code representation and centrality analysis for Scribe.

## Overview

`scribe-graph` builds dependency graphs from repository code and computes centrality metrics to identify the most important files. It implements research-grade graph algorithms specifically adapted for code analysis, enabling Scribe to intelligently prioritize files based on their structural importance.

## Key Features

### Dependency Graph Construction
- **Multi-language import extraction**: Python, JavaScript/TypeScript, Rust, Go, Java
- **Bidirectional edges**: Tracks both imports (who depends on me) and exports (what I depend on)
- **Transitive closure computation**: Find all direct and indirect dependencies
- **Strongly connected components (SCC)**: Detect circular dependency cycles using Kosaraju's algorithm

### PageRank Centrality
- **Research-grade implementation** adapted specifically for code dependency graphs
- **Reverse edge emphasis**: Importance flows TO imported files (not FROM)
- **Configurable damping factor**: Default 0.85 (standard PageRank value)
- **Convergence detection**: L1 norm delta with configurable precision (1e-6 default)
- **Parallel computation**: Multi-core PageRank using Rayon
- **Performance**: 10ms for small graphs, ~100ms for 10k+ node graphs

### Graph Algorithms
- **Betweenness centrality**: Identifies bridge files connecting different parts of the codebase
- **Degree centrality**: Counts import/export connections
- **Community detection**: Groups related files into clusters
- **Distance computation**: BFS-based shortest path for understanding relationships
- **Graph metrics**: Diameter, clustering coefficient, density

### Surgical Covering Set Selection
- **Entity-targeted selection**: Find minimal file sets needed to understand specific functions/classes
- **Transitive dependency/dependent computation**: Configurable depth limits
- **Inclusion reasons**: Explains why each file was selected (target, direct dep, transitive, etc.)
- **Impact analysis mode**: Analyzes what breaks if you change a target

## Architecture

```
FileMetadata → Import Extraction → Dependency Graph → Centrality Analysis → Selection
      ↓              ↓                    ↓                   ↓               ↓
  AST Parse    Language-specific    petgraph DiGraph    PageRank/SCC    Coverage Set
                  Patterns          (Node: File)       Scoring/Ranking   Computation
                                    (Edge: Import)
```

### Core Components

#### `DependencyGraph`
Main graph structure built on `petgraph::DiGraph`:
- **Nodes**: File paths with metadata
- **Edges**: Import relationships with weight
- **Indexing**: Fast lookup by file path using `HashMap`
- **Serialization**: Save/load graphs for caching

#### `ImportExtractor`
Language-specific import parsing:
- **Python**: `import`, `from ... import` statements
- **JavaScript/TypeScript**: `import`, `require()`, ES modules
- **Rust**: `use`, `mod`, `extern crate` declarations
- **Go**: `import` blocks with package resolution
- **Java**: `import` statements with wildcard support

#### `CentralityComputer`
Implements graph centrality algorithms:
- **PageRank**: Iterative power method with convergence checks
- **Betweenness**: Brandes' algorithm for shortest path betweenness
- **Closeness**: Normalized distance to all other nodes
- **Eigenvector**: Spectral centrality via power iteration

#### `CoveringSetSelector`
Computes minimal file sets for entity understanding:
- **Entity search**: AST-based function/class/module lookup with fuzzy matching
- **Closure computation**: Transitive dependencies and dependents
- **Importance filtering**: Excludes noise below threshold
- **Reason tracking**: Records why each file was included

## Usage

### Building a Dependency Graph

```rust
use scribe_graph::{DependencyGraph, ImportExtractor};

let extractor = ImportExtractor::new();
let mut graph = DependencyGraph::new();

for file in files {
    let imports = extractor.extract_imports(&file)?;
    graph.add_node(file.path.clone(), file.clone());

    for import in imports {
        graph.add_edge(file.path.clone(), import.target, import.weight);
    }
}

println!("Graph has {} nodes and {} edges",
    graph.node_count(),
    graph.edge_count()
);
```

### Computing PageRank Centrality

```rust
use scribe_graph::centrality::{PageRankConfig, compute_pagerank};

let config = PageRankConfig {
    damping: 0.85,
    max_iterations: 100,
    tolerance: 1e-6,
    parallel: true,
};

let scores = compute_pagerank(&graph, config)?;

// Get top 10 most central files
let mut ranked: Vec<_> = scores.iter().collect();
ranked.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

for (file, score) in ranked.iter().take(10) {
    println!("{}: {:.6}", file.display(), score);
}
```

### Surgical Covering Set Selection

```rust
use scribe_graph::covering::{CoveringSetConfig, compute_covering_set};

let config = CoveringSetConfig {
    entity_name: "authenticate_user".to_string(),
    entity_type: EntityType::Function,
    max_depth: 3,
    include_dependents: false, // for understanding
    importance_threshold: 0.01,
};

let result = compute_covering_set(&graph, config)?;

println!("Selected {} files to understand function:", result.files.len());
for (file, reason) in result.files {
    println!("  {}: {:?}", file.display(), reason);
}
```

### Transitive Dependencies

```rust
use scribe_graph::traversal::compute_transitive_dependencies;

let target = PathBuf::from("src/auth/login.rs");
let deps = compute_transitive_dependencies(&graph, &target, Some(3))?;

println!("Dependencies of {}: ", target.display());
for (depth, dep) in deps {
    println!("  [depth {}] {}", depth, dep.display());
}
```

## PageRank for Code

Traditional PageRank flows importance FROM pages that link TO you. For code:

**Key Insight**: Important files are those that *many other files depend on*, not files that depend on many things.

**Implementation**: We reverse edge direction when computing PageRank:
- If `A imports B`, we add edge `B → A` in the PageRank graph
- Importance flows FROM importers TO imported files
- Highly imported files (like `utils.rs`, `config.py`) get high scores

**Example**:
```
src/main.rs imports utils/helpers.rs
src/api.rs imports utils/helpers.rs
src/db.rs imports utils/helpers.rs

→ helpers.rs gets high PageRank (many files depend on it)
→ main.rs gets lower PageRank (only helpers depends on it)
```

## Performance

### Targets
- **Small graphs (<100 nodes)**: <10ms PageRank computation
- **Medium graphs (100-1k)**: <50ms PageRank computation
- **Large graphs (1k-10k)**: <200ms PageRank computation
- **Very large (10k+)**: <1s PageRank computation

### Optimizations
- **Sparse computation**: Only iterate over non-zero edges
- **Parallel PageRank**: Multi-threaded score updates using Rayon
- **Early convergence**: Stop iteration when delta falls below tolerance
- **Graph caching**: Serialize computed metrics for reuse
- **Lazy SCC**: Only compute strongly connected components when needed

## Configuration

### `PageRankConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `damping` | `f64` | `0.85` | Damping factor (probability of following links) |
| `max_iterations` | `usize` | `100` | Maximum iterations before stopping |
| `tolerance` | `f64` | `1e-6` | Convergence threshold (L1 norm delta) |
| `parallel` | `bool` | `true` | Use parallel computation |

### `CoveringSetConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `entity_name` | `String` | - | Function/class/module to target |
| `entity_type` | `EntityType` | - | Function, Class, Module, etc. |
| `max_depth` | `Option<usize>` | `None` | Limit transitive dependency depth |
| `include_dependents` | `bool` | `false` | Include files that depend on target |
| `importance_threshold` | `f64` | `0.01` | Exclude files below centrality score |

## Integration

`scribe-graph` is used by:

- **scribe-selection**: Applies PageRank scores to file selection heuristics
- **scribe-scaling**: Uses graph metrics for context positioning optimization
- **scribe-webservice**: Provides API endpoints for graph visualization
- **CLI covering-set command**: Surgical file selection for specific entities

## See Also

- `scribe-analysis`: Provides AST parsing for import extraction
- `scribe-selection`: Uses centrality scores for file ranking
- `scribe-core`: Shared types and configuration
- `../../scribe-rs/scribe-scaling/docs/context-positioning.md`: Context optimization using centrality

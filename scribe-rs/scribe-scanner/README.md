# scribe-scanner

High-performance file system scanning and indexing for Scribe repository analysis.

## Overview

`scribe-scanner` is the foundational crate responsible for efficiently traversing repositories, filtering files, detecting languages, and building the initial file metadata that feeds into Scribe's analysis pipeline. It handles repositories of any size—from small projects to enterprise codebases with 100k+ files.

## Key Features

### Fast Repository Traversal
- **Parallel scanning** using `rayon` for multi-core file system traversal
- **Smart filtering** respects `.gitignore`, `.scribeignore`, and custom patterns
- **Early pruning** of excluded directories to minimize filesystem operations
- **Metadata caching** with xxhash-based content signatures for incremental updates

### Language Detection
- **Extension-based detection** for 20+ programming languages
- **Heuristic analysis** for ambiguous files (e.g., detecting shell scripts without extensions)
- **Language metrics** including LOC, comment density, and complexity estimates
- **Tree-sitter integration** for AST-based language validation

### File Classification
- **ContentType detection**: Code, documentation, configuration, data, test files
- **Template detection**: Identifies boilerplate and auto-generated files
- **Test file recognition**: Links test files to their corresponding source files
- **Documentation scoring**: Prioritizes README, design docs, and API documentation

### Git Integration
- **Commit history analysis** for file churn metrics (change frequency and recency)
- **Blame data extraction** for understanding file evolution
- **Branch awareness** for selective scanning

## Architecture

```
Repository → Scanner → FileMetadata → Analysis Pipeline
   ↓            ↓           ↓
.gitignore   Filter    Language
Patterns    Engine    Detection
            ↓           ↓
        Ignore      AST Parser
        Rules      (tree-sitter)
```

### Core Components

#### `RepositoryScanner`
Main entry point for repository traversal. Configurable scanning options:
- Token budgets and file size limits
- Include/exclude patterns (glob and regex)
- Test file inclusion/exclusion
- Parallel scanning thread count

#### `FileMetadata`
Rich metadata structure containing:
- Path, size, modification time
- Language detection results
- Content type classification
- Git churn metrics
- Token estimates
- Importance scoring hints

#### `IgnoreEngine`
Handles pattern matching for file exclusion:
- `.gitignore` parsing using `ignore` crate
- Custom `.scribeignore` patterns
- Binary file detection and exclusion
- Size-based filtering

#### `LanguageDetector`
Determines file language and characteristics:
- Extension mapping to languages
- Shebang detection for scripts
- Tree-sitter parser availability checking
- Language-specific metrics

## Usage

### Basic Scanning

```rust
use scribe_scanner::{RepositoryScanner, ScanConfig};

let config = ScanConfig {
    root_path: PathBuf::from("."),
    max_file_size: 1_000_000, // 1MB
    exclude_tests: true,
    ..Default::default()
};

let scanner = RepositoryScanner::new(config);
let files = scanner.scan().await?;

println!("Scanned {} files", files.len());
for file in files {
    println!("{}: {} ({})", file.path.display(), file.language, file.size);
}
```

### Custom Patterns

```rust
use scribe_scanner::{ScanConfig, PatternSet};

let mut config = ScanConfig::default();
config.exclude_patterns = PatternSet::new(vec![
    "**/*.log",
    "**/node_modules/**",
    "**/.venv/**",
]);

config.include_patterns = PatternSet::new(vec![
    "src/**/*.rs",
    "lib/**/*.py",
]);

let scanner = RepositoryScanner::new(config);
```

### Git Churn Analysis

```rust
use scribe_scanner::git::ChurnAnalyzer;

let analyzer = ChurnAnalyzer::new(".")?;
let churn_data = analyzer.analyze_file("src/main.rs")?;

println!("Changes: {}", churn_data.commit_count);
println!("Last modified: {}", churn_data.last_change);
println!("Recent activity score: {:.2}", churn_data.recency_score);
```

## Performance

### Targets
- **Small repos (≤1k files)**: <500ms scan time, <50MB memory
- **Medium repos (1k-10k)**: <3s scan time, <200MB memory
- **Large repos (10k-100k)**: <15s scan time, <1GB memory
- **Enterprise (100k+)**: <30s scan time, <2GB memory

### Optimizations
- **Parallel traversal**: Multi-threaded directory walking
- **Early filtering**: Prune excluded directories before scanning
- **Lazy AST parsing**: Only parse files when needed for analysis
- **Incremental caching**: Content-based signatures avoid re-scanning unchanged files
- **String interning**: Reduce memory overhead for repeated path components

## Configuration

### `ScanConfig` Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `root_path` | `PathBuf` | `"."` | Repository root directory |
| `max_file_size` | `usize` | 1MB | Skip files larger than this |
| `exclude_tests` | `bool` | `false` | Exclude test files from scan |
| `follow_symlinks` | `bool` | `false` | Follow symbolic links |
| `include_patterns` | `PatternSet` | Empty | Glob patterns for inclusion |
| `exclude_patterns` | `PatternSet` | Empty | Glob patterns for exclusion |
| `max_depth` | `Option<usize>` | None | Maximum directory depth |
| `parallel_threads` | `usize` | CPU count | Scanner thread pool size |

## Integration

`scribe-scanner` is designed as a foundational crate used by higher-level components:

- **scribe-analysis**: Consumes `FileMetadata` for AST parsing and import extraction
- **scribe-graph**: Uses scan results to build dependency graphs
- **scribe-selection**: Applies scoring to scanned files for selection decisions
- **scribe-scaling**: Optimizes scan performance for large repositories

## See Also

- `scribe-patterns`: Advanced pattern matching and glob support
- `scribe-analysis`: AST parsing and semantic analysis
- `scribe-core`: Shared types and configuration
- `../../ARCHITECTURE.md`: Overall system design

# scribe-scaling

High-performance scaling optimizations for large repository analysis in Scribe.

## Overview

`scribe-scaling` is the performance and optimization layer that enables Scribe to handle repositories of any size—from small projects to enterprise codebases with 100k+ files. It implements streaming architecture, intelligent caching, parallel processing, and context positioning to maintain sub-second performance on small repos and sub-30-second analysis on massive repositories.

## Key Features

### Progressive Loading Architecture
- **Metadata-first streaming**: Load file metadata before content to enable early filtering
- **Lazy content loading**: Only read file content when selected for inclusion
- **Backpressure management**: Adaptive throttling prevents memory exhaustion
- **Incremental processing**: Process files as they're discovered, not after full scan

### Intelligent Caching
- **Persistent cache**: Disk-based cache with Blake3 content signatures
- **Signature-based invalidation**: Only re-process changed files
- **Multi-level caching**: In-memory LRU + disk persistence
- **Compression**: Compressed cache entries using flate2 (gzip)
- **Cache warming**: Pre-populate cache for common operations

### Parallel Processing
- **Multi-core scanning**: Parallel directory traversal using Rayon
- **Concurrent analysis**: Parallel AST parsing and scoring
- **Async I/O**: Non-blocking file operations with tokio
- **Work stealing**: Dynamic load balancing across threads
- **Adaptive parallelism**: Scale thread count based on system resources

### Context Positioning Optimization
- **Transformer-aware ordering**: Exploits LLM attention patterns
- **3-tier positioning**: HEAD (20%) → MIDDLE (60%) → TAIL (20%)
- **Query-aware relevance**: Surfaces most relevant files at HEAD
- **Centrality-based placement**: High-centrality files at HEAD and TAIL
- **Relatedness grouping**: Clusters related files together

### Adaptive Thresholds
- **Repository-aware configuration**: Auto-tune based on repo size
- **Memory-aware limits**: Adjust based on available system memory
- **Performance targets**: Dynamic optimization for time/memory tradeoffs
- **Quality preservation**: Maintain information density while scaling

## Architecture

```
Repository → Streaming Scan → Parallel Analysis → Selection + Positioning → Caching → Output
     ↓              ↓                ↓                      ↓                 ↓          ↓
  Metadata     Progressive      Multi-threaded        3-Tier Context      Blake3    Optimized
   First         Loading          AST/Scoring          HEAD/MIDDLE/TAIL   Signature   Bundle
                ↓                     ↓                                       ↓
            Backpressure         Work Stealing                          LRU + Disk
             Control              Load Balance                            Cache
```

### Core Components

#### `ScalingSelector`
Main entry point for scaled repository processing:
- Integrates all optimization layers
- Configurable performance/quality tradeoffs
- Automatic adaptation based on repository characteristics
- Progress reporting and cancellation support

#### `StreamingScanner`
Progressive file system traversal:
- Yields file metadata as discovered
- Supports filtering during traversal
- Adaptive buffering based on memory pressure
- Early termination support

#### `ParallelAnalyzer`
Multi-threaded file analysis:
- Rayon-based thread pool
- Dynamic work distribution
- Priority-based scheduling (high-importance files first)
- Error isolation (one failure doesn't stop processing)

#### `ContextPositioner`
Optimizes file order for LLM recall:
- **HEAD**: Query-relevant high-centrality files (transformers attend best here)
- **MIDDLE**: Supporting files with lower centrality (background context)
- **TAIL**: Core functionality high-centrality files (attention strengthens again)
- Centrality calculation using PageRank
- Query relevance scoring with term matching
- Relatedness grouping by imports and structure

#### `CacheManager`
Persistent caching system:
- Blake3 content hashing for change detection
- LRU eviction for memory cache
- Compressed disk storage
- Atomic cache updates
- Cache statistics and monitoring

#### `AdaptiveConfig`
Auto-tuning configuration:
- Detects repository size and complexity
- Queries available system memory
- Adjusts thread counts, buffer sizes, cache limits
- Scales thresholds based on detected patterns

## Usage

### Basic Scaled Selection

```rust
use scribe_scaling::{ScalingSelector, ScalingConfig};

let config = ScalingConfig {
    performance_target: PerformanceTarget::Balanced,
    enable_caching: true,
    enable_positioning: true,
    ..Default::default()
};

let selector = ScalingSelector::new(config);
let result = selector.select_and_process(repo_path).await?;

println!("Analyzed {} files in {:?}",
    result.files_processed,
    result.elapsed_time
);
println!("Cache hit rate: {:.1}%", result.cache_stats.hit_rate * 100.0);
```

### Query-Aware Context Positioning

```rust
use scribe_scaling::{ScalingSelector, ContextPositioningConfig};

let mut config = ScalingConfig::default();
config.positioning = ContextPositioningConfig {
    enable_positioning: true,
    head_percentage: 0.20,      // 20% high-priority files at HEAD
    tail_percentage: 0.20,      // 20% core files at TAIL
    centrality_weight: 0.5,
    query_relevance_weight: 0.3,
    relatedness_weight: 0.2,
};

let selector = ScalingSelector::new(config);
let result = selector.select_and_process_with_query(
    repo_path,
    Some("authentication middleware")  // Query hint
).await?;

if result.has_context_positioning() {
    let ordered = result.get_optimally_ordered_files();
    println!("HEAD: {} files", ordered.head.len());
    println!("MIDDLE: {} files", ordered.middle.len());
    println!("TAIL: {} files", ordered.tail.len());
}
```

### Streaming with Progress

```rust
use scribe_scaling::{StreamingScanner, ScanConfig};
use indicatif::{ProgressBar, ProgressStyle};

let scanner = StreamingScanner::new(ScanConfig::default());
let progress = ProgressBar::new_spinner();

progress.set_style(
    ProgressStyle::default_spinner()
        .template("{spinner} [{elapsed}] {msg} ({pos} files)")
);

let mut file_stream = scanner.scan_streaming(repo_path).await?;

while let Some(file) = file_stream.next().await {
    progress.set_message(format!("Scanning {}", file.path.display()));
    progress.inc(1);

    // Process file metadata immediately
    if file.score > threshold {
        // Load and analyze content only for high-scoring files
        let content = file.load_content().await?;
        analyze(content).await?;
    }
}

progress.finish_with_message("Scan complete");
```

### Custom Performance Targets

```rust
use scribe_scaling::{PerformanceTarget, ScalingConfig};

// Fast mode: Prioritize speed over completeness
let fast_config = ScalingConfig {
    performance_target: PerformanceTarget::Speed,
    parallel_threads: Some(num_cpus::get()),
    cache_size_mb: 500,
    max_file_size: 500_000,  // Skip large files
    enable_positioning: false,
};

// Quality mode: Prioritize completeness and quality
let quality_config = ScalingConfig {
    performance_target: PerformanceTarget::Quality,
    parallel_threads: Some(4),  // Fewer threads, more thorough
    cache_size_mb: 2000,
    max_file_size: 5_000_000,
    enable_positioning: true,
};

// Balanced mode (default)
let balanced_config = ScalingConfig::default();
```

### Cache Management

```rust
use scribe_scaling::cache::{CacheManager, CacheConfig};

let cache_config = CacheConfig {
    cache_dir: PathBuf::from(".scribe-cache"),
    max_size_mb: 1000,
    compression_level: 6,
    ttl_hours: 24,
};

let cache = CacheManager::new(cache_config)?;

// Check cache status
let stats = cache.stats();
println!("Cache entries: {}", stats.entry_count);
println!("Total size: {} MB", stats.size_mb);
println!("Hit rate: {:.1}%", stats.hit_rate * 100.0);

// Clear old entries
cache.evict_expired()?;

// Clear entire cache
cache.clear()?;
```

## Performance Targets

### Repository Size Profiles

| Size | Files | Time Target | Memory Target | Strategy |
|------|-------|-------------|---------------|----------|
| **Small** | ≤1k | <1s | <50MB | In-memory, minimal caching |
| **Medium** | 1k-10k | <5s | <200MB | Parallel + caching |
| **Large** | 10k-100k | <15s | <1GB | Streaming + aggressive caching |
| **Enterprise** | 100k+ | <30s | <2GB | Full optimization suite |

### Achieved Performance

Based on internal benchmarks:

- **Linux kernel** (70k files): ~12s analysis time, ~800MB memory
- **Chromium** (300k files): ~28s analysis time, ~1.8GB memory
- **Small Rust project** (500 files): ~450ms analysis time, ~40MB memory

## Context Positioning

### Why It Matters

Transformer models don't attend equally to all tokens:
- **Strong attention at HEAD**: First 20% gets highest attention
- **Weak attention in MIDDLE**: Middle 60% gets less focus
- **Strong attention at TAIL**: Final 20% gets second-highest attention

**Strategy**: Place most important files where LLMs attend best.

### 3-Tier System

#### HEAD (20%)
- Query-relevant files with high centrality
- Entry points matching query terms
- Critical dependencies for query context
- **Goal**: Give LLM immediate relevant context

#### MIDDLE (60%)
- Supporting files with lower centrality
- Utility code and helpers
- Test files (if included)
- **Goal**: Provide background without overwhelming attention

#### TAIL (20%)
- Core high-centrality files (lib.rs, __init__.py)
- Foundational configuration and types
- Architectural anchors
- **Goal**: Ground LLM understanding with core concepts

### Positioning Algorithm

```
1. Compute PageRank centrality for all files
2. Score query relevance (if query provided)
3. Combined score = centrality_weight * centrality +
                    query_weight * relevance
4. Sort by combined score
5. Top 20% → HEAD
6. Middle 60% → MIDDLE
7. Bottom 20% with highest centrality → TAIL
8. Group related files within each tier
```

## Configuration

### `ScalingConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `performance_target` | `PerformanceTarget` | `Balanced` | Speed, Balanced, or Quality |
| `parallel_threads` | `Option<usize>` | CPU count | Thread pool size |
| `cache_size_mb` | `usize` | `1000` | Maximum cache size |
| `enable_caching` | `bool` | `true` | Enable persistent cache |
| `enable_positioning` | `bool` | `true` | Enable context positioning |
| `max_file_size` | `usize` | `1_000_000` | Skip files larger than this |

### `ContextPositioningConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_positioning` | `bool` | `true` | Enable/disable positioning |
| `head_percentage` | `f64` | `0.20` | Percentage for HEAD section |
| `tail_percentage` | `f64` | `0.20` | Percentage for TAIL section |
| `centrality_weight` | `f64` | `0.4` | Weight for centrality scoring |
| `query_relevance_weight` | `f64` | `0.3` | Weight for query matching |
| `relatedness_weight` | `f64` | `0.3` | Weight for file grouping |

## Optimizations

### Memory Management
- **Streaming processing**: Never load entire repository into memory
- **Lazy content loading**: Load file content only when needed
- **Compressed caching**: Reduce cache memory footprint
- **Incremental GC**: Release memory as processing progresses

### I/O Optimization
- **Async file operations**: Non-blocking reads with tokio
- **Read-ahead buffering**: Pre-fetch likely-needed files
- **Memory-mapped files**: For very large files
- **Batched writes**: Coalesce cache updates

### CPU Optimization
- **Work stealing**: Dynamic thread load balancing
- **SIMD**: Use SIMD for hash computation (Blake3)
- **Compiled patterns**: Pre-compile regex and globs
- **Lazy evaluation**: Skip unnecessary computations

## Integration

`scribe-scaling` is used by:

- **CLI**: Top-level orchestration of repository analysis
- **scribe-webservice**: Powers web API for large repositories
- **scribe-selection**: Provides scalable selection infrastructure
- **scribe-graph**: Scales PageRank to large graphs

## See Also

- `docs/context-positioning.md`: Detailed context positioning documentation
- `scribe-selection`: File selection algorithms that scaling optimizes
- `scribe-graph`: PageRank computation used by positioning
- `../../WHY_SCRIBE.md`: Philosophy on performance and intelligence

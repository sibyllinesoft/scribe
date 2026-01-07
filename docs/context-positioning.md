# Context Positioning

Scribe strategically positions files to maximize LLM reasoning quality based on transformer attention patterns.

## The Science

Research shows that transformer models don't attend equally to all tokens in their context window:

- **Strong attention at the beginning** (recency bias)
- **Reduced attention in the middle** (lost-in-the-middle effect)
- **Strong attention at the end** (primacy bias)

Scribe exploits this by positioning files strategically.

## Three-Tier Positioning

### HEAD (20%)

Query-specific, high-centrality files for immediate context:

- Files most relevant to your query
- Entry points and main interfaces
- Types and configurations the target depends on

### MIDDLE (60%)

Supporting context with lower priority:

- Helper utilities and internal functions
- Secondary dependencies
- Less central but still needed files

### TAIL (20%)

Core functionality for foundational understanding:

- `lib.rs`, `main.rs`, `__init__.py` (entry points)
- High-centrality connector files
- Central configuration and types

## How Centrality is Calculated

Scribe uses multiple centrality measures:

### PageRank Centrality

Files heavily referenced by others rank higher:

```
High centrality:
- lib.rs (many files import from it)
- types.rs (defines common types)
- config.rs (widely used)

Low centrality:
- specific_handler.rs (few imports)
- test_utils.rs (only tests import)
```

### Betweenness Centrality

Files that connect different parts of the codebase:

```
High betweenness:
- api/mod.rs (connects handlers to services)
- db/connection.rs (connects app to database)
```

### Degree Centrality

Number of direct import/export connections:

```
High degree:
- prelude.rs (re-exports many items)
- index.ts (barrel file)
```

## Query-Aware Positioning

When you provide a query hint, Scribe scores files by relevance:

```bash
scribe --covering-set "auth" --query "authentication middleware" --stdout
```

Files matching "authentication" or "middleware" get boosted to HEAD:

```
HEAD Section:
  1. auth.rs (centrality: 0.234, relevance: 2.0)
  2. middleware.rs (centrality: 0.156, relevance: 1.5)
  3. session.rs (centrality: 0.189, relevance: 1.0)

MIDDLE Section:
  4. utils/crypto.rs
  5. db/users.rs
  ...

TAIL Section:
  1. lib.rs (centrality: 0.456)
  2. config.rs (centrality: 0.345)
```

## Configuration

```bash
# Custom tier percentages
scribe --head-percentage 0.25 --tail-percentage 0.15 ...

# Adjust weights
scribe --centrality-weight 0.5 \
       --relatedness-weight 0.3 \
       --query-relevance-weight 0.2 ...
```

### Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `head_percentage` | `0.20` | Percentage of files for HEAD |
| `tail_percentage` | `0.20` | Percentage of files for TAIL |
| `centrality_weight` | `0.4` | Weight for centrality scoring |
| `relatedness_weight` | `0.3` | Weight for file grouping |
| `query_relevance_weight` | `0.3` | Weight for query matching |

## Relatedness Grouping

Within each tier, related files are grouped together:

1. **Directory structure**: Files in the same module stay together
2. **Import relationships**: Files that import each other are adjacent
3. **Language similarity**: Same-language files grouped
4. **Functional domains**: API, utils, tests grouped separately

## Example

Given this file selection:

```
Selected files:
- src/api/auth.rs
- src/api/users.rs
- src/db/connection.rs
- src/db/models.rs
- src/lib.rs
- src/config.rs
- src/utils/crypto.rs
- src/utils/validation.rs
- tests/auth_test.rs
```

After context positioning:

```
=== HEAD (query: "authentication") ===
1. src/api/auth.rs        (relevant + moderate centrality)
2. src/utils/crypto.rs    (relevant)

=== MIDDLE ===
3. src/api/users.rs       (related to auth)
4. src/db/models.rs       (supporting)
5. src/db/connection.rs   (supporting)
6. src/utils/validation.rs (supporting)
7. tests/auth_test.rs     (low priority)

=== TAIL ===
8. src/lib.rs             (high centrality entry point)
9. src/config.rs          (high centrality core)
```

## Performance

Context positioning adds minimal overhead:

- **Centrality calculation**: O(n²) where n is selected files
- **Query relevance**: O(n × m) where m is query terms
- **Grouping**: O(n log n) for sorting
- **Overall**: ~0.5-2ms for typical selections

## Benefits

### For LLM Reasoning

- Important context where attention is strongest
- Query-relevant information appears early
- Core functionality provides stable grounding at the end

### For Developers

- Automatic—works without configuration
- Configurable weights and tier sizes
- Explainable positioning decisions

## See Also

- [Covering Sets](covering-sets.md) - How files are selected
- [Architecture](architecture.md) - Implementation details

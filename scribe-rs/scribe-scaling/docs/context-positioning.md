# Context Positioning Optimization

This module implements sophisticated context positioning optimization based on transformer model attention patterns. Models demonstrate better reasoning capabilities at the head and tail of their context window, so we strategically position files to leverage this behavior.

## Overview

The context positioning system analyzes selected files and organizes them into three strategic tiers:

- **HEAD (20%)**: Query-specific high centrality files for immediate context
- **MIDDLE (60%)**: Low centrality supporting files as background
- **TAIL (20%)**: Core functionality, high centrality for foundational understanding

## Key Features

### 1. Centrality Calculation
Files are analyzed using graph-based centrality scoring:
- **Betweenness centrality**: Files that connect different parts of the codebase
- **PageRank centrality**: Files that are heavily referenced by others
- **Degree centrality**: Files with many import/export connections

### 2. Query-Aware Positioning
When provided with a query hint, the system:
- Scores files by relevance to the query terms
- Places most relevant high-centrality files in the HEAD section
- Ensures query-specific entry points appear first in context

### 3. Relatedness Grouping
Files are grouped by:
- Directory structure and module organization
- Import/export relationships
- Language and file type similarities
- Functional domains (API, utils, tests, etc.)

## Usage

### Basic Integration

```rust
use scribe_scaling::{ScalingSelector, ContextPositioningConfig};

// Enable context positioning
let mut config = ScalingSelectionConfig::medium_budget();
config.positioning_config.enable_positioning = true;
let mut selector = ScalingSelector::new(config);

// Use with query hint for optimal positioning
let result = selector.select_and_process_with_query(repo_path, Some("main function")).await?;

if result.has_context_positioning() {
    let ordered_files = result.get_optimally_ordered_files();
    // Files are now in optimal order: HEAD → MIDDLE → TAIL
}
```

### Custom Configuration

```rust
use scribe_scaling::ContextPositioningConfig;

let config = ContextPositioningConfig {
    enable_positioning: true,
    head_percentage: 0.25,          // 25% for HEAD section
    tail_percentage: 0.15,          // 15% for TAIL section  
    centrality_weight: 0.5,         // Higher weight for centrality
    relatedness_weight: 0.3,        // Weight for grouping related files
    query_relevance_weight: 0.2,    // Weight for query matching
};
```

## Architecture

### Positioning Pipeline

```
Selected Files → Centrality Analysis → Query Relevance → Relatedness Grouping → Strategic Positioning
       ↓                ↓                    ↓                    ↓                    ↓
   File Metadata    Graph Analysis      Query Matching      Group Assignment     HEAD/MIDDLE/TAIL
```

### Centrality Analysis

The system builds a dependency graph from file relationships:

```rust
// Example dependency extraction
match file.language.as_str() {
    "Rust" => {
        // lib.rs and mod.rs are central connection points
        if filename != "mod.rs" && filename != "lib.rs" {
            dependencies.push(format!("{}/lib.rs", dir_path));
        }
    }
    "Python" => {
        // __init__.py files are package entry points
        if filename != "__init__.py" {
            dependencies.push(format!("{}/__init__.py", dir_path));
        }
    }
    // ... other languages
}
```

### Query Relevance Scoring

Files are scored based on query term matches:

```rust
for word in query_words {
    if filename.contains(word) {
        relevance += 1.0;  // Exact filename match
    } else if path_str.contains(word) {
        relevance += 0.5;  // Path match
    } else if language.to_lowercase().contains(word) {
        relevance += 0.2;  // Language match
    }
}
```

## Performance Impact

Context positioning adds minimal overhead:
- **Centrality calculation**: O(n²) where n is selected files (typically small)
- **Query relevance**: O(n × m) where m is query terms
- **Grouping**: O(n log n) for sorting within groups
- **Overall**: ~0.5-2ms additional processing time for typical selections

## Benefits

### For Model Reasoning
- **Better attention**: Important files positioned where models focus most
- **Improved context**: Query-relevant information appears early
- **Foundation last**: Core functionality provides stable grounding

### For Developers  
- **Transparent**: Works automatically with existing selection logic
- **Configurable**: Adjustable weights and tier sizes
- **Explainable**: Detailed reasoning provided for positioning decisions

## Examples

### Query-Specific Positioning

```bash
Query: "authentication middleware"

HEAD Section:
  1. auth.rs (centrality: 0.234, relevance: 2.0)
  2. middleware.rs (centrality: 0.156, relevance: 1.5)

TAIL Section: 
  1. lib.rs (centrality: 0.456)
  2. main.rs (centrality: 0.234)
```

### Architectural Understanding

```bash
High centrality files (likely to be in TAIL):
  - lib.rs, main.rs (entry points)
  - mod.rs files (module connectors)  
  - config.rs (widely used utilities)

Low centrality files (likely in MIDDLE):
  - helper utilities
  - specific implementations
  - test files
  - documentation
```

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_positioning` | `true` | Enable/disable context positioning |
| `head_percentage` | `0.20` | Percentage of files for HEAD section |
| `tail_percentage` | `0.20` | Percentage of files for TAIL section |
| `centrality_weight` | `0.4` | Weight for centrality in positioning |
| `relatedness_weight` | `0.3` | Weight for file grouping |
| `query_relevance_weight` | `0.3` | Weight for query matching |

## Integration with Existing Systems

The context positioning system integrates seamlessly:

- **ScalingSelector**: Automatic integration with all selection algorithms
- **Token Management**: Respects existing token budgets and limits
- **Performance**: Maintains sub-second processing times
- **Compatibility**: Optional feature that doesn't break existing workflows

See the `context_positioning_demo.rs` example for a complete demonstration of all features.
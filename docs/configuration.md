# Configuration

Scribe can be configured via CLI flags, environment variables, and configuration files.

## Configuration Precedence

1. CLI flags (highest priority)
2. Environment variables
3. Configuration file
4. Defaults (lowest priority)

## Configuration File

Scribe looks for `.scribe.toml` or `scribe.toml` in the repository root.

### Example Configuration

```toml
# .scribe.toml

[selection]
# Default token budget
token_budget = 100000

# Selection algorithm: simple, complex, heuristic
algorithm = "heuristic"

# PageRank centrality weight (0.0-1.0)
centrality_weight = 0.4

[output]
# Default output format: xml, json, text, markdown
format = "xml"

# Include metadata in output
include_metadata = true

[covering_set]
# Default granularity: entity, file
granularity = "entity"

# Default max depth (0 = unlimited)
max_depth = 0

# Include dependents by default
include_dependents = false

[positioning]
# Enable context positioning
enabled = true

# HEAD section percentage
head_percentage = 0.20

# TAIL section percentage
tail_percentage = 0.20

# Weight for centrality in positioning
centrality_weight = 0.4

# Weight for file relatedness
relatedness_weight = 0.3

# Weight for query relevance
query_relevance_weight = 0.3

[patterns]
# Always include these patterns
include = [
    "src/**",
    "lib/**",
]

# Always exclude these patterns
exclude = [
    "**/node_modules/**",
    "**/__pycache__/**",
    "**/target/**",
    "**/.git/**",
    "**/*.test.*",
    "**/*.spec.*",
]

[git]
# Use git information in analysis
enabled = true

# Prioritize recently modified files
include_recent = true

# How many days counts as "recent"
recent_days = 30
```

## Environment Variables

### Token Budget

```bash
export SCRIBE_TOKEN_BUDGET=100000
```

### Output Format

```bash
export SCRIBE_OUTPUT_FORMAT=xml
```

### Logging

```bash
# Log levels: error, warn, info, debug, trace
export SCRIBE_LOG=info

# Disable colors
export NO_COLOR=1
```

## Selection Configuration

### Token Budget

Controls the maximum tokens in output:

```toml
[selection]
token_budget = 100000
```

```bash
scribe --token-budget 100000
```

### Algorithm

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| `simple` | Size and path depth | Quick analysis |
| `complex` | Multi-dimensional scoring | Thorough analysis |
| `heuristic` | Balanced approach | General use (default) |

```toml
[selection]
algorithm = "heuristic"
```

### Centrality Weight

How much to weight PageRank centrality (0.0-1.0):

- `0.0` - Ignore centrality
- `0.5` - Balanced
- `1.0` - Centrality only

```toml
[selection]
centrality_weight = 0.4
```

## Covering Set Configuration

### Granularity

| Granularity | Description |
|-------------|-------------|
| `entity` | Extract specific functions/classes |
| `file` | Include whole files |

```toml
[covering_set]
granularity = "entity"
```

### Max Depth

Limit dependency traversal:

```toml
[covering_set]
max_depth = 3  # 0 = unlimited
```

## Context Positioning Configuration

### Enable/Disable

```toml
[positioning]
enabled = true
```

### Tier Sizes

```toml
[positioning]
head_percentage = 0.20  # 20% for HEAD
tail_percentage = 0.20  # 20% for TAIL
# MIDDLE gets the remaining 60%
```

### Scoring Weights

```toml
[positioning]
centrality_weight = 0.4       # PageRank importance
relatedness_weight = 0.3      # File grouping
query_relevance_weight = 0.3  # Query matching
```

## Pattern Configuration

### Include Patterns

```toml
[patterns]
include = [
    "src/**",
    "lib/**",
    "app/**",
]
```

### Exclude Patterns

```toml
[patterns]
exclude = [
    "**/node_modules/**",
    "**/vendor/**",
    "**/__pycache__/**",
    "**/target/**",
    "**/dist/**",
    "**/build/**",
    "**/.git/**",
    "**/*.test.*",
    "**/*.spec.*",
    "**/*.min.js",
]
```

## Git Configuration

```toml
[git]
# Enable git awareness
enabled = true

# Weight recent files higher
include_recent = true

# Definition of "recent"
recent_days = 30
```

## Per-Project Overrides

You can have different configurations for different contexts:

```bash
# Development
scribe --config .scribe.dev.toml

# Production/CI
scribe --config .scribe.ci.toml
```

## Preset Configurations

### Minimal Analysis

```toml
[selection]
token_budget = 50000
algorithm = "simple"

[covering_set]
max_depth = 2
granularity = "file"

[positioning]
enabled = false
```

### Deep Analysis

```toml
[selection]
token_budget = 200000
algorithm = "complex"
centrality_weight = 0.6

[covering_set]
max_depth = 0  # Unlimited
granularity = "entity"
include_dependents = true

[positioning]
enabled = true
head_percentage = 0.25
```

### AI Agent Optimized

```toml
[selection]
algorithm = "heuristic"
centrality_weight = 0.5

[output]
format = "xml"
include_metadata = true

[covering_set]
granularity = "entity"
max_depth = 3

[positioning]
enabled = true
```

## See Also

- [CLI Reference](cli-reference.md) - All CLI options
- [Context Positioning](context-positioning.md) - Positioning details

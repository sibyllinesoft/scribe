# CLI Usage

This guide covers common Scribe CLI usage patterns and examples.

## Basic Usage

```bash
# Analyze current directory
scribe

# Analyze specific path
scribe /path/to/repo

# Output to file
scribe --output bundle.md
```

## Covering Set Analysis

The core feature of Scribe is extracting a function/class and all its dependencies.

### Target a Specific Entity

```bash
# Function in a file
scribe --covering-set "src/auth.rs:authenticate_user" --stdout

# Class in a module
scribe --covering-set "api/handlers.py:UserController" --stdout

# Just a file (all entities)
scribe --covering-set "src/lib.rs" --stdout
```

### Control Granularity

```bash
# Entity-level (functions/classes) - more precise
scribe --covering-set "src/auth.rs:login" --granularity entity --stdout

# File-level - faster, less precise
scribe --covering-set "src/auth.rs" --granularity file --stdout
```

### Limit Depth

```bash
# Only direct dependencies
scribe --covering-set "main.rs:main" --max-depth 1 --stdout

# Up to 3 levels of dependencies
scribe --covering-set "main.rs:main" --max-depth 3 --stdout
```

### Include Dependents (Impact Analysis)

```bash
# See what depends on this function
scribe --covering-set "src/utils.rs:format_date" --include-dependents --stdout
```

### Git Diff Analysis

```bash
# Analyze current uncommitted changes
scribe --covering-set-diff --stdout

# Useful for code review context
scribe --covering-set-diff --output review-context.md
```

## Repository Bundling

Generate bundles of your entire repository with intelligent selection.

### Output Formats

```bash
# Markdown (default)
scribe --style markdown --output bundle.md

# HTML with interactive editor
scribe --style html --editor --output bundle.html

# XML (for agents)
scribe --output-format xml --output bundle.xml

# JSON (for programmatic use)
scribe --output-format json --output bundle.json
```

### Token Budgeting

```bash
# Stay within token limit
scribe --token-budget 100000 --output bundle.md

# Smaller budget with progressive demotion
scribe --token-budget 50000 --output bundle.md
```

### File Selection

```bash
# Include only specific patterns
scribe --include "src/**" --include "lib/**" --output bundle.md

# Exclude patterns
scribe --exclude "**/*.test.*" --exclude "**/node_modules/**" --output bundle.md

# Both
scribe --include "src/**" --exclude "**/*.test.*" --output bundle.md
```

### Centrality Weighting

```bash
# Higher weight on PageRank centrality
scribe --centrality-weight 0.5 --output bundle.md

# Focus on architectural core
scribe --centrality-weight 0.7 --output architecture.md
```

## Output to Stdout

Use `--stdout` for piping to other tools:

```bash
# Pipe to clipboard (macOS)
scribe --covering-set "main.rs" --stdout | pbcopy

# Pipe to file
scribe --covering-set "main.rs" --stdout > context.txt

# Pipe to another tool
scribe --covering-set "main.rs" --stdout | wc -l
```

## Combining Options

```bash
# Code review workflow
scribe --covering-set-diff \
       --include-dependents \
       --max-files 30 \
       --output-format xml \
       --stdout

# Architecture documentation
scribe --include "src/**" \
       --exclude "**/*.test.*" \
       --centrality-weight 0.6 \
       --token-budget 80000 \
       --style markdown \
       --output architecture.md

# Quick function understanding
scribe --covering-set "api/routes.py:create_order" \
       --max-depth 2 \
       --output-format text \
       --stdout
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SCRIBE_TOKEN_BUDGET` | Default token budget |
| `SCRIBE_OUTPUT_FORMAT` | Default output format |
| `NO_COLOR` | Disable colored output |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Parse error |

## See Also

- [CLI Reference](cli-reference.md) - Complete option listing
- [Covering Sets](covering-sets.md) - Deep dive into dependency analysis
- [Output Formats](output-formats.md) - Format specifications

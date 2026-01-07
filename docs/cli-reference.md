# CLI Reference

Complete reference for all Scribe CLI options.

## Synopsis

```
scribe [OPTIONS] [PATH]
```

## Arguments

### `[PATH]`

Repository path to analyze. Defaults to current directory.

```bash
scribe                    # Current directory
scribe /path/to/repo      # Specific path
```

## Covering Set Options

### `--covering-set <TARGET>`

Compute covering set for a file or entity.

```bash
# File only
--covering-set "src/lib.rs"

# File:entity
--covering-set "src/auth.rs:authenticate_user"

# Class
--covering-set "api/models.py:UserModel"
```

### `--covering-set-diff`

Compute covering set for current git diff.

```bash
--covering-set-diff
```

### `--granularity <MODE>`

Selection granularity.

| Value | Description |
|-------|-------------|
| `entity` | Function/class level (default) |
| `file` | Whole file level |

```bash
--granularity entity
--granularity file
```

### `--include-dependents`

Include files that depend on the target (impact analysis).

```bash
--covering-set "utils.rs:helper" --include-dependents
```

### `--max-depth <N>`

Maximum dependency traversal depth.

```bash
--max-depth 1     # Direct dependencies only
--max-depth 3     # Up to 3 levels
```

### `--max-files <N>`

Maximum files in result.

```bash
--max-files 20
```

## Output Options

### `--output <PATH>` / `-o <PATH>`

Write output to file instead of stdout.

```bash
-o bundle.md
--output bundle.xml
```

### `--stdout`

Force output to stdout (useful with --output-format).

```bash
--covering-set "main.rs" --stdout
```

### `--output-format <FORMAT>`

Output format for covering sets and bundles.

| Value | Description |
|-------|-------------|
| `xml` | Structured XML (recommended for agents) |
| `json` | JSON format |
| `text` | Plain text |
| `markdown` | Markdown format |

```bash
--output-format xml
--output-format json
```

### `--style <STYLE>`

Output style for repository bundles.

| Value | Description |
|-------|-------------|
| `markdown` | Markdown with code blocks |
| `html` | HTML document |
| `xml` | XML structure |
| `json` | JSON format |
| `repomix` | Repomix-compatible XML |

```bash
--style markdown
--style html --editor
```

### `--editor`

Include interactive editor in HTML output.

```bash
--style html --editor -o bundle.html
```

## Selection Options

### `--token-budget <N>` / `--token-target <N>`

Target token count for selection.

```bash
--token-budget 100000
--token-target 50000
```

### `--include <PATTERN>`

Include only files matching pattern(s). Can be repeated.

```bash
--include "src/**"
--include "src/**" --include "lib/**"
```

### `--exclude <PATTERN>`

Exclude files matching pattern(s). Can be repeated.

```bash
--exclude "**/*.test.*"
--exclude "**/node_modules/**" --exclude "**/__pycache__/**"
```

### `--algorithm <ALGO>`

Selection algorithm.

| Value | Description |
|-------|-------------|
| `simple` | Basic heuristics |
| `complex` | Multi-dimensional scoring |
| `heuristic` | Balanced approach (default) |

```bash
--algorithm heuristic
```

### `--centrality-weight <WEIGHT>`

Weight for PageRank centrality in scoring (0.0-1.0).

```bash
--centrality-weight 0.5
```

## Context Positioning Options

### `--enable-positioning`

Enable context positioning optimization (default: true).

```bash
--enable-positioning
--no-positioning
```

### `--head-percentage <PERCENT>`

Percentage of files for HEAD section (0.0-1.0).

```bash
--head-percentage 0.25
```

### `--tail-percentage <PERCENT>`

Percentage of files for TAIL section (0.0-1.0).

```bash
--tail-percentage 0.15
```

### `--query <QUERY>`

Query hint for relevance-based positioning.

```bash
--query "authentication middleware"
```

## Git Options

### `--git-aware`

Use git information in analysis.

```bash
--git-aware
```

### `--include-recent`

Prioritize recently modified files.

```bash
--include-recent
```

## General Options

### `--help` / `-h`

Show help message.

```bash
scribe --help
scribe --covering-set --help
```

### `--version` / `-V`

Show version.

```bash
scribe --version
```

### `--verbose` / `-v`

Increase verbosity. Can be repeated.

```bash
-v        # Info level
-vv       # Debug level
-vvv      # Trace level
```

### `--quiet` / `-q`

Suppress non-essential output.

```bash
-q
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SCRIBE_TOKEN_BUDGET` | Default token budget | 128000 |
| `SCRIBE_OUTPUT_FORMAT` | Default output format | text |
| `NO_COLOR` | Disable colored output | - |
| `SCRIBE_LOG` | Log level | warn |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File/path not found |
| 4 | Parse error |
| 5 | Git error |

## Examples

### Basic Covering Set

```bash
scribe --covering-set "src/main.rs:main" --stdout
```

### Code Review Context

```bash
scribe --covering-set-diff \
       --include-dependents \
       --max-files 30 \
       --output-format xml \
       --stdout
```

### Architecture Documentation

```bash
scribe --include "src/**" \
       --exclude "**/*.test.*" \
       --centrality-weight 0.6 \
       --token-budget 80000 \
       --style markdown \
       --output architecture.md
```

### Interactive Bundle Editor

```bash
scribe --style html \
       --editor \
       --token-budget 100000 \
       --output bundle.html
```

### AI Agent Integration

```bash
scribe --covering-set "api/routes.py:create_order" \
       --max-depth 3 \
       --output-format xml \
       --stdout
```

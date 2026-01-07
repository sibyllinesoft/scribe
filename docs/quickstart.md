# Quickstart

Get started with Scribe in under 5 minutes.

## Installation

### npm (Recommended)

```bash
# Install globally
npm install -g @sibyllinesoft/scribe

# Or use directly with npx
npx @sibyllinesoft/scribe --help
```

### Cargo

```bash
# From crates.io
cargo install scribe-cli

# From source
git clone https://github.com/sibyllinesoft/scribe
cd scribe
cargo install --path scribe-rs --locked
```

## Your First Covering Set

The most powerful feature of Scribe is **covering set analysis**—getting a function and all its dependencies in one call.

```bash
# Navigate to your repository
cd your-project

# Get a function and all its dependencies
scribe --covering-set "src/auth.rs:authenticate_user" --stdout
```

This returns the target function plus every type, function, and constant it needs—automatically traced through imports.

## Common Workflows

### Understand a Specific Function

```bash
# Get the function with all dependencies, limit depth to 3
scribe --covering-set "src/api/handlers.py:create_user" \
       --max-depth 3 \
       --stdout
```

### Analyze Git Changes

```bash
# Get context for your current git diff
scribe --covering-set-diff --stdout
```

### Generate Repository Bundle

```bash
# Create a Markdown bundle with intelligent file selection
scribe --style markdown --output bundle.md

# Stay within token budget
scribe --token-budget 100000 --style markdown --output bundle.md
```

### Interactive HTML Editor

```bash
# Create an interactive editor to review and customize
scribe --style html --editor --output bundle.html
```

## Output Formats

Scribe supports multiple output formats:

```bash
# XML (recommended for agents - structured with metadata)
scribe --covering-set "module.py:MyClass" --stdout --output-format xml

# JSON (for programmatic use)
scribe --covering-set "module.py:MyClass" --stdout --output-format json

# Plain text (human readable)
scribe --covering-set "module.py:MyClass" --stdout --output-format text
```

## Next Steps

- [CLI Usage](cli-usage.md) - Learn all CLI options
- [Covering Sets](covering-sets.md) - Deep dive into dependency analysis
- [Why Scribe](why-scribe.md) - Understand the design philosophy

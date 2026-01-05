# @sibyllinesoft/scribe

Intelligent code context for AI agents - surgical code retrieval with dependency resolution.

## Installation

```bash
npm install -g @sibyllinesoft/scribe
# or
npx @sibyllinesoft/scribe --help
```

## Usage

```bash
# Get a function and all its dependencies
scribe --covering-set "src/auth.rs:authenticate_user" --stdout

# Get file-level dependencies
scribe --covering-set "src/auth.rs" --granularity file --stdout

# Analyze code affected by current changes
scribe --covering-set-diff --stdout
```

## Why Scribe?

Unlike tools that bundle entire repositories, Scribe understands your code's dependency graph and returns only what's needed:

| Approach | Tool Calls | Tokens Used | Relevance |
|----------|------------|-------------|-----------|
| Manual grep + read | 4-10+ | ~15,000 | ~5% relevant |
| Full repo bundle | 1 | ~500,000 | ~1% relevant |
| **Scribe** | **1** | **~2,000** | **95%+ relevant** |

## Supported Platforms

- macOS (Apple Silicon & Intel)
- Linux (x64 & arm64)
- Windows (x64)

## Documentation

Full documentation: https://github.com/sibyllinesoft/scribe

## License

MIT OR Apache-2.0

# Scribe

**Intelligent Code Context for AI Agents**

Scribe is a code analysis tool designed for AI agents and LLM-powered development workflows. Unlike simple file bundlers, Scribe understands code structure and dependencies—giving agents exactly the context they need without wasting tokens on irrelevant code.

## The Problem

When an AI agent needs to understand a function and its dependencies, existing approaches are painful:

| Approach | Tool Calls | Tokens Used | Relevance |
|----------|------------|-------------|-----------|
| Manual grep + read | 4-10+ | ~15,000 | ~5% relevant |
| LSP iterative lookup | 5-15+ | ~8,000 | ~40% relevant |
| Full repo bundlers | 1 | ~500,000 | ~1% relevant |
| **Scribe covering set** | **1** | **~2,000** | **95%+ relevant** |

## The Solution

**Scribe provides transitive context expansion.** Ask for a function and get everything in its dependency cone—types, called functions, constants, configs—in a single call.

```bash
# Get a function and all its dependencies
scribe --covering-set "src/auth.rs:authenticate_user" --stdout
```

Returns:
```
- authenticate_user (target)
- verify_password, create_session (direct dependencies)
- PasswordHash, Session, AuthConfig (type dependencies)
- AUTH_TIMEOUT constant (implicit dependency)
```

**One call. Complete context. No iterative discovery.**

## Key Features

- **Covering Set Analysis**: Get specific functions/classes with all their dependencies
- **Entity-Level Granularity**: Extract exactly what you need, not entire files
- **Multi-Language Support**: Rust, Python, JavaScript/TypeScript, Go, Elixir
- **PageRank Centrality**: Graph-based importance ranking
- **Token Budget Management**: Progressive demotion within any context limit
- **Git Integration**: Analyze changes and their impact

## Quick Links

| | |
|---|---|
| **[Quickstart](quickstart.md)** | Install Scribe and run your first covering set analysis |
| **[Why Scribe](why-scribe.md)** | Understand covering sets, context positioning, and more |
| **[CLI Reference](cli-reference.md)** | Complete command-line interface documentation |
| **[Architecture](architecture.md)** | Learn about Scribe's internal design |

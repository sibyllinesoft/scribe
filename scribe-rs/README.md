# Scribe - Intelligent Code Context for AI Agents

[![Crates.io](https://img.shields.io/crates/v/scribe.svg)](https://crates.io/crates/scribe)
[![Documentation](https://docs.rs/scribe/badge.svg)](https://docs.rs/scribe)
[![License](https://img.shields.io/crates/l/scribe.svg)](https://github.com/sibyllinesoft/scribe#license)
[![Build Status](https://github.com/sibyllinesoft/scribe/workflows/CI/badge.svg)](https://github.com/sibyllinesoft/scribe/actions)

Scribe is a code analysis tool designed for AI agents and LLM-powered development workflows. Unlike simple file bundlers, Scribe understands code structure and dependencies—giving agents exactly the context they need without wasting tokens on irrelevant code.

## The Problem: Context Retrieval is Expensive

When an AI agent needs to understand a function and its dependencies, the traditional approach is painful:

```
Agent wants to understand `authenticate_user()` in auth.rs

┌─────────────────────────────────────────────────────────────────────────────┐
│  TRADITIONAL APPROACH (4-10+ tool calls, ~30 seconds, wastes tokens)        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. grep "authenticate_user" --include="*.rs"     → Find the function      │
│  2. read auth.rs                                  → Read ENTIRE 800-line file│
│  3. grep "use crate::" auth.rs                    → Find imports manually  │
│  4. read session.rs                               → Read ENTIRE dependency │
│  5. grep "use crate::" session.rs                 → Find transitive imports│
│  6. read crypto.rs                                → Another full file read │
│  7. read config.rs                                → Keep going...          │
│  ...                                                                        │
│                                                                             │
│  Result: Agent reads 4000+ lines, but only ~200 are relevant               │
│  Cost: Multiple round-trips, 95% wasted tokens, slow iteration             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  SCRIBE APPROACH (1 tool call, ~0.7 seconds, precise context)               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  $ scribe --covering-set "auth.rs:authenticate_user" --stdout              │
│                                                                             │
│  Result: Returns authenticate_user + only the functions/types it uses       │
│  - auth.rs:authenticate_user (target)                                       │
│  - session.rs:create_session (direct dependency)                           │
│  - crypto.rs:verify_password (direct dependency)                           │
│  - config.rs:AuthConfig (type dependency)                                  │
│                                                                             │
│  Cost: Single call, ~200 lines of precisely relevant code                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Scribe's covering set feature understands your code's dependency graph and returns only what's needed.**

## Key Differentiator: Surgical Code Retrieval

Unlike tools like repomix that bundle entire repositories, Scribe provides **surgical precision**:

| Approach | Tool Calls | Tokens Used | Relevance |
|----------|------------|-------------|-----------|
| Manual grep + read | 4-10+ | ~15,000 | ~5% relevant |
| Repomix (full bundle) | 1 | ~500,000 | ~1% relevant |
| **Scribe covering set** | **1** | **~2,000** | **95%+ relevant** |

This matters because:
- **Faster iteration**: Single call vs. multiple round-trips
- **Lower cost**: 10-100x fewer tokens per context retrieval
- **Better results**: LLMs perform better with focused, relevant context
- **Automatic dependency resolution**: No manual import tracing

## Quick Start

### For AI Agents (CLI)

```bash
# Get a function and all its dependencies
scribe --covering-set "src/auth.rs:authenticate_user" --stdout

# Get file-level dependencies (faster, less precise)
scribe --covering-set "src/auth.rs" --granularity file --stdout

# Analyze what code is affected by your current changes
scribe --covering-set-diff --stdout

# Limit depth for focused context
scribe --covering-set "src/lib.rs:Config" --max-depth 2 --stdout
```

### Output Formats

```bash
# XML output (recommended for agents - structured, includes metadata)
scribe --covering-set "module.py:MyClass" --stdout --output-format xml

# JSON output (for programmatic use)
scribe --covering-set "module.py:MyClass" --stdout --output-format json

# Plain text (human readable)
scribe --covering-set "module.py:MyClass" --stdout --output-format text
```

### Example Output

```xml
<?xml version="1.0" encoding="UTF-8"?>
<covering_set>
  <files count="3">
    <file>
      <path>src/auth.rs</path>
      <distance>0</distance>
      <reason>TargetFile</reason>
      <content><![CDATA[
pub fn authenticate_user(credentials: &Credentials) -> Result<Session> {
    let user = lookup_user(&credentials.username)?;
    verify_password(&credentials.password, &user.password_hash)?;
    create_session(user.id)
}
]]></content>
    </file>
    <file>
      <path>src/session.rs</path>
      <distance>1</distance>
      <reason>DirectDependency</reason>
      <content><![CDATA[
pub fn create_session(user_id: UserId) -> Result<Session> {
    // ... only the relevant function, not the whole file
}
]]></content>
    </file>
  </files>
  <statistics>
    <files_examined>142</files_examined>
    <files_selected>3</files_selected>
    <max_depth_reached>2</max_depth_reached>
  </statistics>
</covering_set>
```

## Features

### Covering Set Analysis
- **Entity-level granularity**: Get specific functions/classes, not entire files
- **Automatic dependency resolution**: Follows imports across your codebase
- **Multi-language support**: Rust, Python, JavaScript/TypeScript, Go
- **Configurable depth**: Control how deep to traverse dependencies
- **Diff-based analysis**: Get context for your current git changes

### Repository Bundling
- **Intelligent file selection**: PageRank-based importance scoring
- **Token budget management**: Stay within LLM context limits
- **Multiple output formats**: HTML, XML, JSON, Markdown, Repomix-compatible

### Code Analysis
- **Dependency graph construction**: Understand code relationships
- **Heuristic scoring**: Identify important files automatically
- **Git integration**: Incorporate change history into analysis

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
cargo install --path .
```

## Supported Languages

Import resolution and dependency tracking works for:

| Language | Import Styles Supported |
|----------|------------------------|
| **Rust** | `use`, `mod`, grouped imports `use mod::{a, b}` |
| **Python** | `import`, `from...import`, relative imports |
| **JavaScript/TypeScript** | ES6 `import`, `require()`, type imports |
| **Go** | Single imports, block imports, aliased imports |

## CLI Reference

### Covering Set Options

```
--covering-set <TARGET>     Find covering set for file or entity
                            Examples: "src/lib.rs", "src/auth.rs:login"

--covering-set-diff         Compute covering set for current git diff

--granularity <MODE>        file (whole files) or entity (functions/classes)
                            Default: file

--include-dependents        Include files that depend on target (impact analysis)

--max-depth <N>             Maximum dependency traversal depth

--max-files <N>             Maximum files in result

--stdout                    Output to stdout (for piping to other tools)

--output-format <FMT>       xml, json, text, markdown
```

### Repository Bundling Options

```
--token-target <N>          Target token count for selection (default: 128000)

--include <PATTERNS>        Include only matching files

--exclude <PATTERNS>        Exclude matching files

--output-format <FMT>       html, xml, json, text, markdown, repomix
```

## Library Usage

Scribe can also be used as a Rust library:

```rust
use scribe::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Analyze a repository
    let config = Config::default();
    let analysis = analyze_repository(".", &config).await?;

    // Get most important files
    for (file, score) in analysis.top_files(10) {
        println!("{}: {:.3}", file, score);
    }

    Ok(())
}
```

### Feature Flags

```toml
[dependencies]
# Full installation (default)
scribe = "0.5"

# Minimal - core types only
scribe = { version = "0.5", default-features = false, features = ["core"] }

# Analysis without graph features
scribe = { version = "0.5", default-features = false, features = ["core", "analysis", "scanner"] }
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          scribe-cli                             │
├─────────────────────────────────────────────────────────────────┤
│  scribe (main library)                                          │
│  ┌─────────────┐ ┌───────────────┐ ┌─────────────────────────┐  │
│  │ scribe-core │ │scribe-scanner │ │   scribe-patterns       │  │
│  │  (types,    │ │ (file system  │ │ (glob, gitignore)       │  │
│  │  config)    │ │  traversal)   │ │                         │  │
│  └─────────────┘ └───────────────┘ └─────────────────────────┘  │
│  ┌─────────────┐ ┌───────────────┐ ┌─────────────────────────┐  │
│  │  scribe-    │ │ scribe-graph  │ │   scribe-selection      │  │
│  │  analysis   │ │  (PageRank,   │ │  (covering sets,        │  │
│  │ (heuristics)│ │  dependency   │ │   token budgeting)      │  │
│  │             │ │   graphs)     │ │                         │  │
│  └─────────────┘ └───────────────┘ └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Performance

- **Covering set computation**: ~0.7s for 140-file codebase
- **Full repository analysis**: ~100ms for small repos, ~1-10s for large repos
- **Memory usage**: ~2MB per 1000 files

## Comparison with Other Tools

| Feature | Scribe | Repomix | Manual |
|---------|--------|---------|--------|
| Dependency-aware selection | ✅ | ❌ | ❌ |
| Entity-level granularity | ✅ | ❌ | ❌ |
| Single-command context | ✅ | ✅ | ❌ |
| Token-efficient output | ✅ | ❌ | ❌ |
| Multi-language support | ✅ | ✅ | N/A |
| Git diff analysis | ✅ | ❌ | ❌ |

## License

Licensed under either of Apache License 2.0 or MIT license at your option.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

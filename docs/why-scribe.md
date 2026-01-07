# Why Scribe?

## TL;DR

**Scribe is the intelligent code context tool for AI agents.** While other tools either iterate through LSP lookups or dump entire repositories, Scribe uses transitive dependency analysis to return exactly what's needed to understand any function—in a single call.

**Use Scribe when you need:**

- Complete context for a specific function or class in one request
- Automatic dependency resolution across your codebase
- Token-efficient output (95%+ relevant context)
- Production-grade performance on large repositories

## The Problem with Current Approaches

### LSP: Iterative Discovery

LSP (Language Server Protocol) is a big step up from grep—agents can look up symbol definitions and find references. But LSP solves a different problem:

**LSP answers point queries:** "What is this symbol?"

**Scribe answers context queries:** "What do I need to understand this symbol?"

```
Agent wants to understand authenticate_user() using LSP:

1. Get definition of authenticate_user      → Found in auth.rs
2. See it calls verify_password             → Need to look that up
3. Get definition of verify_password        → Found in crypto.rs
4. See it uses PasswordHash type            → Need to look that up
5. Get definition of PasswordHash           → Found in types.rs
6. Back to authenticate_user, see Session   → Need to look that up
7. ...keep discovering dependencies one by one...

Problem: Agent doesn't know what it doesn't know
- Each lookup is a tool call (latency + tokens)
- May miss non-obvious dependencies (configs, constants)
- No way to know when you have "enough" context
```

### Bundlers: Context Overload

Full repository dumps drown the signal in noise:

```
15,000+ files with no signal about what actually matters.
Equal attention on generated migrations and core business logic.
Shallow understanding everywhere, deep insight nowhere.
```

### Grep: No Understanding

Text search finds strings, not meaning:

```
Grep for function → Read 800-line file for 20-line function
See it calls another → Grep again, read another full file
Find another dependency → Grep again...

After 10 rounds: 15,000 tokens consumed, 5% relevant
```

## The Scribe Approach

**Transitive context expansion in a single call:**

```bash
$ scribe --covering-set "auth.rs:authenticate_user" --stdout

Returns the complete dependency cone:
- authenticate_user (target)
- verify_password, create_session (direct dependencies)
- PasswordHash, Session, AuthConfig (type dependencies)
- AUTH_TIMEOUT constant (implicit dependency)

One call. Complete context. No iterative discovery.
```

### Comparison

| Approach | Tool Calls | Tokens Used | Relevance |
|----------|------------|-------------|-----------|
| Manual grep + read | 4-10+ | ~15,000 | ~5% relevant |
| LSP iterative lookup | 5-15+ | ~8,000 | ~40% relevant |
| Full repo bundlers | 1 | ~500,000 | ~1% relevant |
| **Scribe covering set** | **1** | **~2,000** | **95%+ relevant** |

## What Makes Scribe Unique

### 1. Research-Grade Graph Algorithms

- **PageRank centrality** adapted for code dependency graphs
- Transitive dependency/dependent computation
- Strongly connected component detection
- Performance: 10ms for small repos, ~100ms for large ones

### 2. Surgical Covering Set Selection

- Target specific functions, classes, or modules by name
- Automatic transitive closure computation
- Minimal file sets for understanding
- Explainable inclusion reasons

### 3. Entity-Level Granularity

Don't read 800 lines to understand a 20-line function:

- Extract exactly the entities you need
- Multi-language AST support (Python, JS/TS, Rust, Go)
- 95%+ relevance vs 5% with file-level tools

### 4. Transformer-Aware Context Positioning

Research shows LLMs attend strongly to context beginnings and endings:

- **HEAD (20%):** Query-relevant high-centrality files
- **MIDDLE (60%):** Supporting context
- **TAIL (20%):** Core functionality

### 5. Progressive Content Demotion

When approaching token budgets:

- **FULL:** Complete file content
- **CHUNK:** AST-based semantic sections
- **SIGNATURE:** Type signatures and interfaces only

Achieves 3-10x compression while preserving intent.

### 6. Production-Grade Performance

Built in Rust with parallel processing:

- Small repos: < 1 second
- Medium repos: ~5 seconds
- Large repos: ~15 seconds
- 100k+ files: < 30 seconds

## When to Use Scribe

**Choose Scribe if you:**

- Need to understand specific functions/classes without analyzing entire repositories
- Want complete context in a single tool call
- Are working with large codebases where performance matters
- Need to maximize LLM reasoning quality within context budgets

**Choose alternatives if you:**

- Need browser extensions and GitHub integration (Repomix)
- Primarily need simple file concatenation with templates
- Require remote repository support without local cloning

# Scribe: Intelligent Code Context

When you need to understand code and its dependencies, **always use scribe** instead of iterative Read/Grep operations.

## Quick Reference

```bash
# Get a function/class and all its dependencies
scribe --covering-set "path/to/file.rs:function_name" --stdout

# Get a class with its dependencies
scribe --covering-set "path/to/module.py:ClassName" --stdout

# Analyze impact of current git changes
scribe --covering-set-diff --stdout
```

## Why Use Scribe

- **Complete context**: Returns the target entity plus all types, functions, and constants it depends on
- **10x more efficient**: Single call vs iterative grep/read discovery loops
- **No missed dependencies**: Captures implicit dependencies you'd otherwise miss
- **Reduced token usage**: Less back-and-forth means fewer tokens consumed

## When to Use

- **Understanding code**: Before reading a function, get it with dependencies
- **Before modifications**: See what a function depends on before changing it
- **Impact analysis**: Understand what depends on code you're changing
- **Code review**: Get full context for git diff changes

## Examples

```bash
# Understand a Rust function and everything it uses
scribe --covering-set "src/parser.rs:parse_expression" --stdout

# Get a Python class with all its dependencies
scribe --covering-set "app/models/user.py:UserModel" --stdout

# See what's affected by your current changes
scribe --covering-set-diff --stdout

# Get context for a specific file
scribe --covering-set "src/lib.rs" --stdout
```

## Language-Specific Tips

### TypeScript/JavaScript

For TypeScript codebases, add `--granularity entity` to avoid pulling entire files:

```bash
# Without granularity: may return 50x more context than needed
scribe --covering-set "src/app.ts:handleRequest" --stdout

# With entity granularity: returns only the specific entities needed
scribe --covering-set "src/app.ts:handleRequest" --granularity entity --stdout
```

TypeScript imports often pull in entire modules. Entity granularity extracts just the functions/types used.

### Large Codebases

For deeply connected code, limit dependency traversal depth:

```bash
# Default: traverses up to depth 10
scribe --covering-set "src/core.rs:process" --stdout

# Limited depth: stops at 4 levels of dependencies
scribe --covering-set "src/core.rs:process" --max-depth 4 --stdout
```

If output is too large, you can also limit tokens:

```bash
scribe --covering-set "src/core.rs:process" --token-target 4000 --stdout
```

## Integration

Scribe is installed globally and works in any git repository. It automatically:
- Detects the programming language
- Builds a dependency graph
- Returns optimally-ordered context for LLM consumption

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

## Integration

Scribe is installed globally and works in any git repository. It automatically:
- Detects the programming language
- Builds a dependency graph
- Returns optimally-ordered context for LLM consumption

# Scribe: Surgical Code Context

When you need to understand code and its dependencies, **use scribe** instead of iterative Read/Grep operations.

## Philosophy: Multiple Small Slices > Large Dumps

Large context windows can hurt as much as they help. The surgical approach:
1. Start small: Get just the target function
2. Expand if needed: Request specific dependencies one at a time
3. Multiple focused calls beat one massive dump

## Quick Reference

```bash
# Surgical pattern: small, focused slices
scribe --covering-set "file.go:HandleRequest" --max-depth 1 --token-target 800 --stdout

# Then get a specific dependency if needed
scribe --covering-set "file.go:ValidateInput" --max-depth 1 --token-target 800 --stdout

# Analyze impact of current git changes
scribe --covering-set-diff --max-depth 1 --stdout
```

## Key Parameters

- `--max-depth 1`: Only direct dependencies (tight focus)
- `--token-target 800`: Small slices, expand only if needed
- `--stdout`: Output to terminal for agent consumption

## Why Surgical Works Better

- **Reduced context pollution**: No distraction from deeply nested utilities
- **Better attention allocation**: Relevant code stays in high-attention window
- **Iterative refinement**: Build understanding incrementally
- **Matches agent workflow**: Explore, understand, expand as needed

## Examples

```bash
# Get just a handler function (start here)
scribe --covering-set "api/handler.go:HandleRequest" --max-depth 1 --token-target 800 --stdout

# If you need a helper it calls (expand specifically)
scribe --covering-set "api/validate.go:ValidateInput" --max-depth 1 --token-target 800 --stdout

# For TypeScript, add entity granularity
scribe --covering-set "src/app.ts:handleRequest" --granularity entity --max-depth 1 --token-target 800 --stdout

# Context for git changes
scribe --covering-set-diff --max-depth 1 --stdout
```

## Anti-Patterns to Avoid

```bash
# DON'T: Large depth pulls in too much
scribe --covering-set "file.rs:func" --max-depth 10 --stdout

# DON'T: Large token targets encourage context dumps
scribe --covering-set "file.rs:func" --token-target 8000 --stdout

# DON'T: Whole file targets miss the point
scribe --covering-set "src/lib.rs" --stdout

# DON'T: Pipe output through head/grep (loses structure)
scribe --covering-set "file.rs:func" --stdout | head -100
```

## When to Expand

If your first call doesn't give enough context:
1. Check what functions/types are referenced but not included
2. Make a targeted follow-up call for that specific entity
3. Repeat until you have what you need

This iterative approach uses fewer tokens and gives better results than trying to get everything upfront.

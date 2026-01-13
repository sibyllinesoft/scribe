# Claude Code vs Scribe Benchmark Results

**Generated:** 2026-01-10T09:01:46.459823
**Model:** haiku
**Targets:** 3
**Total Runs:** 3

## Executive Summary

| Metric | Value |
|--------|-------|
| **Mean Token Ratio** | 0.2x |
| **Token Ratio Range** | 0.1x - 0.3x |
| **Total Cost** | $0.3755 |

### Key Finding

> **Claude Code uses 0.2x more tokens** than scribe's covering-set output
> to accomplish the same code understanding task.

## Per-Target Results

| Target | Claude Tokens | Scribe Tokens | Ratio | Cost |
|--------|---------------|---------------|-------|------|
| ast_parse_chunks | 2,502 | 8,643 | 0.3x | $0.1120 |
| token_counter_count | 1,462 | 6,259 | 0.2x | $0.1772 |
| centrality_calculate | 2,401 | 27,467 | 0.1x | $0.0863 |

## Methodology

- **Claude Code**: Run with `--print --output-format json` to capture exact token usage
- **Scribe**: Single `--covering-set` call returning dependency context
- **Token comparison**: Claude's input+output tokens vs scribe's output size (chars/4)

## What This Measures

Claude Code was given this task for each target:
> Find all dependencies for function X in file Y. Read the target file, trace dependencies,
> and provide a summary of all direct and transitive dependencies.

This measures the **real cost** of iterative code discovery vs dependency-aware retrieval.

## Raw Data

See `claude_code_benchmark_20260110_090146.json` for complete run data.

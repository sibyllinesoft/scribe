# Claude Code vs Scribe Benchmark Results

**Generated:** 2026-01-10T12:11:04.425333
**Model:** haiku
**Targets:** 3
**Total Runs:** 3

## Executive Summary

| Metric | Value |
|--------|-------|
| **Mean Context Ratio** | 28.6x |
| **Context Ratio Range** | 13.6x - 41.7x |
| **Average Turns** | 13.0 |
| **Total Cost** | $1.0130 |

### Key Finding

> **Claude Code processes 28.6x more context** than scribe's covering-set output
> to accomplish the same code understanding task, across 13.0 turns on average.

## Per-Target Results

| Target | Claude Context | Scribe Output | Ratio | Cost | Turns |
|--------|----------------|---------------|-------|------|-------|
| token_counter_count | 261,018 | 6,259 | 41.7x | $0.3673 | 9.0 |
| ast_parse_chunks | 263,985 | 8,643 | 30.5x | $0.2510 | 10.0 |
| centrality_calculate | 373,819 | 27,467 | 13.6x | $0.3947 | 20.0 |


## Methodology

- **Claude Code**: Run with `--print --output-format json` to capture exact token usage
- **Scribe**: Single `--covering-set` call returning dependency context
- **Context comparison**: Claude's full context (including cached reads) vs scribe's output

## What This Measures

Claude Code was given this task for each target:
> Find all dependencies for function X in file Y. Read the target file, trace dependencies,
> and provide a summary of all direct and transitive dependencies.

The **context ratio** shows how much more context Claude processes during multi-turn discovery
compared to scribe's single-call output. Most of Claude's context is redundant re-reading
of files across turns (cached, but still counted).

## Interpretation

- **Higher ratio** = more redundant context processing
- **More turns** = more iterative discovery steps
- Scribe provides the same understanding in **1 call** with **minimal context**

## Raw Data

See `claude_code_benchmark_20260110_121104.json` for complete run data.

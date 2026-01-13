# Claude Code vs Scribe Benchmark Results

**Generated:** 2026-01-10T12:16:36.801446
**Model:** haiku
**Targets:** 3
**Total Runs:** 9

## Executive Summary

| Metric | Value |
|--------|-------|
| **Mean Context Ratio** | 32.1x |
| **Context Ratio Range** | 17.3x - 46.0x |
| **Average Turns** | 13.0 |
| **Total Cost** | $0.3668 |

### Key Finding

> **Claude Code processes 32.1x more context** than scribe's covering-set output
> to accomplish the same code understanding task, across 13.0 turns on average.

## Per-Target Results

| Target | Claude Context | Scribe Output | Ratio | Cost | Turns |
|--------|----------------|---------------|-------|------|-------|
| ast_parse_chunks | 397,927 | 8,643 | 46.0x | $0.1249 | 12.7 |
| token_counter_count | 206,933 | 6,259 | 33.1x | $0.0766 | 7.7 |
| centrality_calculate | 475,223 | 27,467 | 17.3x | $0.1653 | 18.7 |


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

See `claude_code_benchmark_20260110_121636.json` for complete run data.

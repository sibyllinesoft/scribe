# Claude Code vs Scribe Benchmark Results

**Generated:** 2026-01-10T09:07:47.514677
**Model:** haiku
**Targets:** 3
**Total Runs:** 3

## Executive Summary

| Metric | Value |
|--------|-------|
| **Mean Context Ratio** | 42.4x |
| **Context Ratio Range** | 16.0x - 71.0x |
| **Average Turns** | 14.3 |
| **Total Cost** | $0.3664 |

### Key Finding

> **Claude Code processes 42.4x more context** than scribe's covering-set output
> to accomplish the same code understanding task, across 14.3 turns on average.

## Per-Target Results

| Target | Claude Context | Scribe Output | Ratio | Cost | Turns |
|--------|----------------|---------------|-------|------|-------|
| ast_parse_chunks | 613,865 | 8,643 | 71.0x | $0.1629 | 20.0 |
| token_counter_count | 251,210 | 6,259 | 40.1x | $0.0889 | 9.0 |
| centrality_calculate | 439,296 | 27,467 | 16.0x | $0.1146 | 14.0 |

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

See `claude_code_benchmark_20260110_090747.json` for complete run data.

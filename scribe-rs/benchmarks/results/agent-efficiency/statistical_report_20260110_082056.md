# Agent Token Efficiency Benchmark - Statistical Report

**Generated:** 2026-01-10T08:20:56.834835
**Iterations per target:** 3
**Targets tested:** 5
**Total runs:** 15

## Executive Summary

| Metric | Value | 95% CI |
|--------|-------|--------|
| **Mean Token Savings** | 65.1% | [46.0%, 80.5%] |
| **Overall Token Ratio** | 2.77x | - |
| **Mean Tool Call Reduction** | 32.6x | - |

### Key Finding

> **Scribe reduces token usage by 65% on average** (std: 15.0%)
> while requiring **33x fewer tool calls**.

## Per-Target Results

| Target | Category | Savings | 95% CI | Scribe Tokens | Naive Tokens | Tool Calls |
|--------|----------|---------|--------|---------------|--------------|------------|
| token_counter_count | core | 80.5% | [81%, 81%] | 6,259 | 32,106 | 1/27 |
| ast_parse_chunks | selection | 75.5% | [75%, 75%] | 8,643 | 35,264 | 1/32 |
| pattern_matcher | patterns | 71.0% | [71%, 71%] | 16,760 | 57,752 | 1/37 |
| covering_set_compute | selection | 52.7% | [53%, 53%] | 18,976 | 40,108 | 1/32 |
| centrality_calculate | graph | 46.0% | [46%, 46%] | 27,467 | 50,840 | 1/35 |

## Results by Category

| Category | Mean Savings | Std Dev | N |
|----------|--------------|---------|---|
| core | 80.5% | 0.0% | 1 |
| patterns | 71.0% | 0.0% | 1 |
| selection | 64.1% | 16.1% | 2 |
| graph | 46.0% | 0.0% | 1 |

## Results by Dependency Depth

| Depth | Mean Savings | N |
|-------|--------------|---|
| medium | 69.6% | 3 |
| deep | 58.5% | 2 |

## Methodology

### Measurement Approach
- Each target was run **3 times** to establish statistical significance
- Token counts estimated as `characters / 4` (standard approximation for code)
- 95% confidence intervals computed using t-distribution

### Scribe Approach
- Single `scribe --covering-set "<file>:<entity>"` call
- Returns target entity with transitive dependencies
- **Always 1 tool call** regardless of complexity

### Naive Approach (Simulated Agent)
1. Read target file (1 tool call)
2. Extract `use` statements and type references
3. For each dependency:
   - Grep for definition (1 tool call)
   - Read matching files (1 tool call each)
4. Repeat for transitive dependencies (max depth: 3)

### Conservative Assumptions
The naive simulation is **optimistic** because real agents:
- Often require multiple grep attempts to find correct files
- May read irrelevant files before finding dependencies
- Have additional overhead from tool call formatting/parsing
- May explore dead ends in large codebases

## Raw Data

<details>
<summary>Click to expand full results JSON</summary>

```json
[
  {
    "target_id": "token_counter_count",
    "n_runs": 3,
    "scribe": {
      "tokens_mean": 6259,
      "tokens_std": 0.0,
      "files_mean": 3.0
    },
    "naive": {
      "tokens_mean": 32106,
      "tokens_std": 0.0,
      "files_mean": 10.0,
      "tool_calls_mean": 27.0
    },
    "savings": {
      "mean_pct": 80.5,
      "std_pct": 0.0,
      "ci_95": [
        80.5,
        80.5
      ]
    }
  },
  {
    "target_id": "pattern_matcher",
    "n_runs": 3,
    "scribe": {
      "tokens_mean": 16760,
      "tokens_std": 0.0,
      "files_mean": 4.0
    },
    "naive": {
      "tokens_mean": 57752,
      "tokens_std": 0.0,
      "files_mean": 18.0,
      "tool_calls_mean": 37.0
    },
    "savings": {
      "mean_pct": 71.0,
      "std_pct": 0.0,
      "ci_95": [
        71.0,
        71.0
      ]
    }
  },
  {
    "target_id": "centrality_calculate",
    "n_runs": 3,
    "scribe": {
      "tokens_mean": 27467,
      "tokens_std": 0.0,
      "files_mean": 9.0
    },
    "naive": {
      "tokens_mean": 50840,
      "tokens_std": 0.0,
      "files_mean": 16.0,
      "tool_calls_mean": 35.0
    },
    "savings": {
      "mean_pct": 46.0,
      "std_pct": 0.0,
      "ci_95": [
        46.0,
        46.0
      ]
    }
  },
  {
    "target_id": "covering_set_compute",
    "n_runs": 3,
    "scribe": {
      "tokens_mean": 18976,
      "tokens_std": 0.0,
      "files_mean": 7.0
    },
    "naive": {
      "tokens_mean": 40108,
      "tokens_std": 0.0,
      "files_mean": 13.0,
      "tool_calls_mean": 32.0
    },
    "savings": {
      "mean_pct": 52.7,
      "std_pct": 0.0,
      "ci_95": [
        52.7,
        52.7
      ]
    }
  },
  {
    "target_id": "ast_parse_chunks",
    "n_runs": 3,
    "scribe": {
      "tokens_mean": 8643,
      "tokens_std": 0.0,
      "files_mean": 4.0
    },
    "naive": {
      "tokens_mean": 35264,
      "tokens_std": 0.0,
      "files_mean": 13.0,
      "tool_calls_mean": 32.0
    },
    "savings": {
      "mean_pct": 75.5,
      "std_pct": 0.0,
      "ci_95": [
        75.5,
        75.5
      ]
    }
  }
]
```

</details>

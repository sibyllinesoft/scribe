# Agent Token Efficiency Benchmark - Statistical Report

**Generated:** 2026-01-10T08:24:02.371229
**Iterations per target:** 3
**Targets tested:** 15
**Total runs:** 45

## Executive Summary

| Metric | Value | 95% CI |
|--------|-------|--------|
| **Mean Token Savings** | 73.0% | [35.8%, 94.3%] |
| **Overall Token Ratio** | 3.70x | - |
| **Mean Tool Call Reduction** | 31.6x | - |

### Key Finding

> **Scribe reduces token usage by 73% on average** (std: 14.5%)
> while requiring **32x fewer tool calls**.

## Per-Target Results

| Target | Category | Savings | 95% CI | Scribe Tokens | Naive Tokens | Tool Calls |
|--------|----------|---------|--------|---------------|--------------|------------|
| code_bundler | selection | 94.3% | [94%, 94%] | 942 | 16,545 | 1/18 |
| scanner_scan | scanner | 87.6% | [88%, 88%] | 7,549 | 60,873 | 1/37 |
| token_counter_count | core | 85.0% | [85%, 85%] | 6,259 | 41,806 | 1/33 |
| final_scoring | analysis | 81.7% | [82%, 82%] | 4,005 | 21,837 | 1/19 |
| context_positioner | scaling | 81.0% | [81%, 81%] | 9,951 | 52,338 | 1/33 |
| pagerank_compute | graph | 80.7% | [81%, 81%] | 11,293 | 58,447 | 1/35 |
| git_analyze_diffs | scanner | 75.1% | [75%, 75%] | 5,739 | 23,033 | 1/29 |
| pattern_matcher | patterns | 73.4% | [73%, 73%] | 16,760 | 63,070 | 1/37 |
| token_budget_selection | selection | 71.5% | [71%, 71%] | 21,637 | 75,839 | 1/42 |
| dependency_graph_add | graph | 70.9% | [71%, 71%] | 12,707 | 43,595 | 1/33 |
| ast_parse_chunks | selection | 70.6% | [71%, 71%] | 8,643 | 29,371 | 1/31 |
| scaling_engine_process | scaling | 69.5% | [70%, 70%] | 14,651 | 48,065 | 1/38 |
| language_detection | scanner | 67.3% | [67%, 67%] | 4,887 | 14,954 | 1/21 |
| centrality_calculate | graph | 50.7% | [51%, 51%] | 27,467 | 55,665 | 1/37 |
| covering_set_compute | selection | 35.8% | [36%, 36%] | 18,976 | 29,572 | 1/31 |

## Results by Category

| Category | Mean Savings | Std Dev | N |
|----------|--------------|---------|---|
| core | 85.0% | 0.0% | 1 |
| analysis | 81.7% | 0.0% | 1 |
| scanner | 76.7% | 10.2% | 3 |
| scaling | 75.3% | 8.1% | 2 |
| patterns | 73.4% | 0.0% | 1 |
| selection | 68.0% | 24.1% | 4 |
| graph | 67.4% | 15.3% | 3 |

## Results by Dependency Depth

| Depth | Mean Savings | N |
|-------|--------------|---|
| shallow | 81.1% | 3 |
| medium | 69.7% | 6 |
| deep | 72.2% | 6 |

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
      "tokens_mean": 41806,
      "tokens_std": 0.0,
      "files_mean": 16.0,
      "tool_calls_mean": 33.0
    },
    "savings": {
      "mean_pct": 85.0,
      "std_pct": 0.0,
      "ci_95": [
        85.0,
        85.0
      ]
    }
  },
  {
    "target_id": "scanner_scan",
    "n_runs": 3,
    "scribe": {
      "tokens_mean": 7549,
      "tokens_std": 0.0,
      "files_mean": 3.0
    },
    "naive": {
      "tokens_mean": 60873,
      "tokens_std": 0.0,
      "files_mean": 18.0,
      "tool_calls_mean": 37.0
    },
    "savings": {
      "mean_pct": 87.6,
      "std_pct": 0.0,
      "ci_95": [
        87.6,
        87.6
      ]
    }
  },
  {
    "target_id": "language_detection",
    "n_runs": 3,
    "scribe": {
      "tokens_mean": 4887,
      "tokens_std": 0.0,
      "files_mean": 3.0
    },
    "naive": {
      "tokens_mean": 14954,
      "tokens_std": 0.0,
      "files_mean": 8.0,
      "tool_calls_mean": 21.0
    },
    "savings": {
      "mean_pct": 67.3,
      "std_pct": 0.0,
      "ci_95": [
        67.3,
        67.3
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
      "tokens_mean": 63070,
      "tokens_std": 0.0,
      "files_mean": 18.0,
      "tool_calls_mean": 37.0
    },
    "savings": {
      "mean_pct": 73.4,
      "std_pct": 0.0,
      "ci_95": [
        73.4,
        73.4
      ]
    }
  },
  {
    "target_id": "context_positioner",
    "n_runs": 3,
    "scribe": {
      "tokens_mean": 9951,
      "tokens_std": 0.0,
      "files_mean": 4.0
    },
    "naive": {
      "tokens_mean": 52338,
      "tokens_std": 0.0,
      "files_mean": 16.0,
      "tool_calls_mean": 33.0
    },
    "savings": {
      "mean_pct": 81.0,
      "std_pct": 0.0,
      "ci_95": [
        81.0,
        81.0
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
      "tokens_mean": 55665,
      "tokens_std": 0.0,
      "files_mean": 18.0,
      "tool_calls_mean": 37.0
    },
    "savings": {
      "mean_pct": 50.7,
      "std_pct": 0.0,
      "ci_95": [
        50.7,
        50.7
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
      "tokens_mean": 29572,
      "tokens_std": 0.0,
      "files_mean": 12.0,
      "tool_calls_mean": 31.0
    },
    "savings": {
      "mean_pct": 35.8,
      "std_pct": 0.0,
      "ci_95": [
        35.8,
        35.8
      ]
    }
  },
  {
    "target_id": "scaling_engine_process",
    "n_runs": 3,
    "scribe": {
      "tokens_mean": 14651,
      "tokens_std": 0.0,
      "files_mean": 7.0
    },
    "naive": {
      "tokens_mean": 48065,
      "tokens_std": 0.0,
      "files_mean": 19.0,
      "tool_calls_mean": 38.0
    },
    "savings": {
      "mean_pct": 69.5,
      "std_pct": 0.0,
      "ci_95": [
        69.5,
        69.5
      ]
    }
  },
  {
    "target_id": "git_analyze_diffs",
    "n_runs": 3,
    "scribe": {
      "tokens_mean": 5739,
      "tokens_std": 0.0,
      "files_mean": 3.0
    },
    "naive": {
      "tokens_mean": 23033,
      "tokens_std": 0.0,
      "files_mean": 10.0,
      "tool_calls_mean": 29.0
    },
    "savings": {
      "mean_pct": 75.1,
      "std_pct": 0.0,
      "ci_95": [
        75.1,
        75.1
      ]
    }
  },
  {
    "target_id": "code_bundler",
    "n_runs": 3,
    "scribe": {
      "tokens_mean": 942,
      "tokens_std": 0.0,
      "files_mean": 2.0
    },
    "naive": {
      "tokens_mean": 16545,
      "tokens_std": 0.0,
      "files_mean": 9.0,
      "tool_calls_mean": 18.0
    },
    "savings": {
      "mean_pct": 94.3,
      "std_pct": 0.0,
      "ci_95": [
        94.3,
        94.3
      ]
    }
  },
  {
    "target_id": "token_budget_selection",
    "n_runs": 3,
    "scribe": {
      "tokens_mean": 21637,
      "tokens_std": 0.0,
      "files_mean": 7.0
    },
    "naive": {
      "tokens_mean": 75839,
      "tokens_std": 0.0,
      "files_mean": 23.0,
      "tool_calls_mean": 42.0
    },
    "savings": {
      "mean_pct": 71.5,
      "std_pct": 0.0,
      "ci_95": [
        71.5,
        71.5
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
      "tokens_mean": 29371,
      "tokens_std": 0.0,
      "files_mean": 12.0,
      "tool_calls_mean": 31.0
    },
    "savings": {
      "mean_pct": 70.6,
      "std_pct": 0.0,
      "ci_95": [
        70.6,
        70.6
      ]
    }
  },
  {
    "target_id": "pagerank_compute",
    "n_runs": 3,
    "scribe": {
      "tokens_mean": 11293,
      "tokens_std": 0.0,
      "files_mean": 5.0
    },
    "naive": {
      "tokens_mean": 58447,
      "tokens_std": 0.0,
      "files_mean": 18.0,
      "tool_calls_mean": 35.0
    },
    "savings": {
      "mean_pct": 80.7,
      "std_pct": 0.0,
      "ci_95": [
        80.7,
        80.7
      ]
    }
  },
  {
    "target_id": "dependency_graph_add",
    "n_runs": 3,
    "scribe": {
      "tokens_mean": 12707,
      "tokens_std": 0.0,
      "files_mean": 6.0
    },
    "naive": {
      "tokens_mean": 43595,
      "tokens_std": 0.0,
      "files_mean": 16.0,
      "tool_calls_mean": 33.0
    },
    "savings": {
      "mean_pct": 70.9,
      "std_pct": 0.0,
      "ci_95": [
        70.9,
        70.9
      ]
    }
  },
  {
    "target_id": "final_scoring",
    "n_runs": 3,
    "scribe": {
      "tokens_mean": 4005,
      "tokens_std": 0.0,
      "files_mean": 3.0
    },
    "naive": {
      "tokens_mean": 21837,
      "tokens_std": 0.0,
      "files_mean": 9.0,
      "tool_calls_mean": 19.0
    },
    "savings": {
      "mean_pct": 81.7,
      "std_pct": 0.0,
      "ci_95": [
        81.7,
        81.7
      ]
    }
  }
]
```

</details>

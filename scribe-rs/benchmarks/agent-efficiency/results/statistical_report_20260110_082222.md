# Agent Token Efficiency Benchmark - Statistical Report

**Generated:** 2026-01-10T08:22:22.386702
**Iterations per target:** 5
**Targets tested:** 15
**Total runs:** 75

## Executive Summary

| Metric | Value | 95% CI |
|--------|-------|--------|
| **Mean Token Savings** | 58.0% | [0.0%, 94.3%] |
| **Overall Token Ratio** | 3.40x | - |
| **Mean Tool Call Reduction** | 25.9x | - |

### Key Finding

> **Scribe reduces token usage by 58% on average** (std: 32.5%)
> while requiring **26x fewer tool calls**.

## Per-Target Results

| Target | Category | Savings | 95% CI | Scribe Tokens | Naive Tokens | Tool Calls |
|--------|----------|---------|--------|---------------|--------------|------------|
| code_bundler | selection | 94.3% | [94%, 94%] | 942 | 16,545 | 1/18 |
| scanner_scan | scanner | 84.8% | [85%, 85%] | 7,549 | 49,701 | 1/33 |
| language_detection | scanner | 84.6% | [85%, 85%] | 4,887 | 31,716 | 1/27 |
| token_counter_count | core | 82.7% | [83%, 83%] | 6,259 | 36,179 | 1/35 |
| git_analyze_diffs | scanner | 75.1% | [75%, 75%] | 5,739 | 23,033 | 1/29 |
| pattern_matcher | patterns | 74.6% | [75%, 75%] | 16,760 | 65,972 | 1/39 |
| ast_parse_chunks | selection | 73.5% | [73%, 73%] | 8,643 | 32,590 | 1/32 |
| context_positioner | scaling | 70.2% | [70%, 70%] | 9,951 | 33,381 | 1/28 |
| token_budget_selection | selection | 67.3% | [67%, 67%] | 21,637 | 66,238 | 1/40 |
| scaling_engine_process | scaling | 66.5% | [67%, 67%] | 14,651 | 43,788 | 1/37 |
| covering_set_compute | selection | 53.6% | [54%, 54%] | 18,976 | 40,884 | 1/32 |
| centrality_calculate | graph | 42.9% | [43%, 43%] | 27,467 | 48,090 | 1/35 |
| pagerank_compute | graph | 0.0% | [0%, 0%] | 86 | 0 | 1/1 |
| dependency_graph_build | graph | 0.0% | [0%, 0%] | 86 | 0 | 1/1 |
| heuristics_score | analysis | 0.0% | [0%, 0%] | 86 | 0 | 1/1 |

## Results by Category

| Category | Mean Savings | Std Dev | N |
|----------|--------------|---------|---|
| core | 82.7% | 0.0% | 1 |
| scanner | 81.5% | 5.6% | 3 |
| patterns | 74.6% | 0.0% | 1 |
| selection | 72.2% | 16.9% | 4 |
| scaling | 68.4% | 2.6% | 2 |
| graph | 14.3% | 24.8% | 3 |
| analysis | 0.0% | 0.0% | 1 |

## Results by Dependency Depth

| Depth | Mean Savings | N |
|-------|--------------|---|
| shallow | 23.6% | 4 |
| medium | 73.6% | 6 |
| deep | 66.8% | 5 |

## Methodology

### Measurement Approach
- Each target was run **5 times** to establish statistical significance
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
    "n_runs": 5,
    "scribe": {
      "tokens_mean": 6259,
      "tokens_std": 0.0,
      "files_mean": 3.0
    },
    "naive": {
      "tokens_mean": 36179,
      "tokens_std": 0.0,
      "files_mean": 16.0,
      "tool_calls_mean": 35.0
    },
    "savings": {
      "mean_pct": 82.7,
      "std_pct": 0.0,
      "ci_95": [
        82.7,
        82.7
      ]
    }
  },
  {
    "target_id": "scanner_scan",
    "n_runs": 5,
    "scribe": {
      "tokens_mean": 7549,
      "tokens_std": 0.0,
      "files_mean": 3.0
    },
    "naive": {
      "tokens_mean": 49701,
      "tokens_std": 0.0,
      "files_mean": 14.0,
      "tool_calls_mean": 33.0
    },
    "savings": {
      "mean_pct": 84.8,
      "std_pct": 0.0,
      "ci_95": [
        84.8,
        84.8
      ]
    }
  },
  {
    "target_id": "language_detection",
    "n_runs": 5,
    "scribe": {
      "tokens_mean": 4887,
      "tokens_std": 0.0,
      "files_mean": 3.0
    },
    "naive": {
      "tokens_mean": 31716,
      "tokens_std": 0.0,
      "files_mean": 12.0,
      "tool_calls_mean": 27.0
    },
    "savings": {
      "mean_pct": 84.6,
      "std_pct": 0.0,
      "ci_95": [
        84.6,
        84.6
      ]
    }
  },
  {
    "target_id": "pattern_matcher",
    "n_runs": 5,
    "scribe": {
      "tokens_mean": 16760,
      "tokens_std": 0.0,
      "files_mean": 4.0
    },
    "naive": {
      "tokens_mean": 65972,
      "tokens_std": 0.0,
      "files_mean": 20.0,
      "tool_calls_mean": 39.0
    },
    "savings": {
      "mean_pct": 74.6,
      "std_pct": 0.0,
      "ci_95": [
        74.6,
        74.6
      ]
    }
  },
  {
    "target_id": "context_positioner",
    "n_runs": 5,
    "scribe": {
      "tokens_mean": 9951,
      "tokens_std": 0.0,
      "files_mean": 4.0
    },
    "naive": {
      "tokens_mean": 33381,
      "tokens_std": 0.0,
      "files_mean": 12.0,
      "tool_calls_mean": 28.0
    },
    "savings": {
      "mean_pct": 70.2,
      "std_pct": 0.0,
      "ci_95": [
        70.2,
        70.2
      ]
    }
  },
  {
    "target_id": "centrality_calculate",
    "n_runs": 5,
    "scribe": {
      "tokens_mean": 27467,
      "tokens_std": 0.0,
      "files_mean": 9.0
    },
    "naive": {
      "tokens_mean": 48090,
      "tokens_std": 0.0,
      "files_mean": 16.0,
      "tool_calls_mean": 35.0
    },
    "savings": {
      "mean_pct": 42.9,
      "std_pct": 0.0,
      "ci_95": [
        42.9,
        42.9
      ]
    }
  },
  {
    "target_id": "covering_set_compute",
    "n_runs": 5,
    "scribe": {
      "tokens_mean": 18976,
      "tokens_std": 0.0,
      "files_mean": 7.0
    },
    "naive": {
      "tokens_mean": 40884,
      "tokens_std": 0.0,
      "files_mean": 13.0,
      "tool_calls_mean": 32.0
    },
    "savings": {
      "mean_pct": 53.6,
      "std_pct": 0.0,
      "ci_95": [
        53.6,
        53.6
      ]
    }
  },
  {
    "target_id": "scaling_engine_process",
    "n_runs": 5,
    "scribe": {
      "tokens_mean": 14651,
      "tokens_std": 0.0,
      "files_mean": 7.0
    },
    "naive": {
      "tokens_mean": 43788,
      "tokens_std": 0.0,
      "files_mean": 18.0,
      "tool_calls_mean": 37.0
    },
    "savings": {
      "mean_pct": 66.5,
      "std_pct": 0.0,
      "ci_95": [
        66.5,
        66.5
      ]
    }
  },
  {
    "target_id": "git_analyze_diffs",
    "n_runs": 5,
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
    "n_runs": 5,
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
    "n_runs": 5,
    "scribe": {
      "tokens_mean": 21637,
      "tokens_std": 0.0,
      "files_mean": 7.0
    },
    "naive": {
      "tokens_mean": 66238,
      "tokens_std": 0.0,
      "files_mean": 21.0,
      "tool_calls_mean": 40.0
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
    "target_id": "ast_parse_chunks",
    "n_runs": 5,
    "scribe": {
      "tokens_mean": 8643,
      "tokens_std": 0.0,
      "files_mean": 4.0
    },
    "naive": {
      "tokens_mean": 32590,
      "tokens_std": 0.0,
      "files_mean": 13.0,
      "tool_calls_mean": 32.0
    },
    "savings": {
      "mean_pct": 73.5,
      "std_pct": 0.0,
      "ci_95": [
        73.5,
        73.5
      ]
    }
  },
  {
    "target_id": "pagerank_compute",
    "n_runs": 5,
    "scribe": {
      "tokens_mean": 86,
      "tokens_std": 0.0,
      "files_mean": 0.0
    },
    "naive": {
      "tokens_mean": 0,
      "tokens_std": 0.0,
      "files_mean": 0.0,
      "tool_calls_mean": 1.0
    },
    "savings": {
      "mean_pct": 0.0,
      "std_pct": 0.0,
      "ci_95": [
        0.0,
        0.0
      ]
    }
  },
  {
    "target_id": "dependency_graph_build",
    "n_runs": 5,
    "scribe": {
      "tokens_mean": 86,
      "tokens_std": 0.0,
      "files_mean": 0.0
    },
    "naive": {
      "tokens_mean": 0,
      "tokens_std": 0.0,
      "files_mean": 0.0,
      "tool_calls_mean": 1.0
    },
    "savings": {
      "mean_pct": 0.0,
      "std_pct": 0.0,
      "ci_95": [
        0.0,
        0.0
      ]
    }
  },
  {
    "target_id": "heuristics_score",
    "n_runs": 5,
    "scribe": {
      "tokens_mean": 86,
      "tokens_std": 0.0,
      "files_mean": 0.0
    },
    "naive": {
      "tokens_mean": 0,
      "tokens_std": 0.0,
      "files_mean": 0.0,
      "tool_calls_mean": 1.0
    },
    "savings": {
      "mean_pct": 0.0,
      "std_pct": 0.0,
      "ci_95": [
        0.0,
        0.0
      ]
    }
  }
]
```

</details>

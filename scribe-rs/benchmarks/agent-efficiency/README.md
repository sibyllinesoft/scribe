# Agent Token Efficiency Benchmark

Measures token usage when AI agents understand code dependencies with vs without scribe.

## Quick Start

```bash
# Run REAL Claude Code benchmark (recommended)
./claude_code_benchmark.py --quick --model haiku     # 1 iteration, 3 targets (cheapest)
./claude_code_benchmark.py -n 3                      # 3 iterations, all targets

# Run simulated agent benchmark (no API key required)
./statistical_benchmark.py              # 5 iterations, all targets
./statistical_benchmark.py --quick      # 3 iterations, subset of targets
```

## What This Measures

### With Scribe
```bash
scribe --covering-set "path/to/file.rs:function_name" --stdout
```
- **1 tool call** returns the function and all transitive dependencies
- Output contains only relevant code

### Without Scribe (Claude Code)
Standard Claude Code behavior - reads files, greps for type definitions:
- Multiple turns to discover dependencies iteratively
- Each turn re-sends conversation history (cached but counted)
- Grep may find false positives (files that mention a type, not define it)
- Reads entire files even when only small parts are needed

### Comparison
| Approach | Tool Calls | Precision | Context Efficiency |
|----------|------------|-----------|-------------------|
| Scribe | 1 | High | Minimal - only relevant code |
| Claude grep/read | 10-30+ | Low | 30-70x more context processed |

## Statistical Rigor

The `statistical_benchmark.py` provides:
- **Multiple iterations** per target for consistency validation
- **95% confidence intervals** using t-distribution
- **Per-category analysis** (core, graph, selection, etc.)
- **Per-depth analysis** (shallow, medium, deep dependencies)
- **Raw JSON data** for further analysis
- **Visualization charts** (requires matplotlib)

## Benchmark Targets

15 targets covering 6 categories:

| Category | Targets | Typical Savings |
|----------|---------|-----------------|
| core | token counting | 80%+ |
| patterns | glob/gitignore matching | 70%+ |
| selection | AST parsing, bundling | 50-75% |
| graph | PageRank, centrality | 45-50% |
| scaling | context positioning | 60%+ |
| scanner | file scanning, git analysis | 50-70% |

## Adding New Targets

Edit `targets.json`:

```json
{
  "id": "my_target",
  "name": "MyModule::my_function",
  "scribe_query": "path/to/file.rs:my_function",
  "category": "selection",
  "expected_depth": "deep"
}
```

Good targets have:
- Dependencies spread across multiple files/crates
- Target function is a small portion of a large file
- 3+ levels of transitive dependencies

## Results

Results are saved to `../results/agent-efficiency/`:
- `statistical_report_*.md` - Full statistical analysis
- `raw_data_*.json` - Machine-readable data
- `token_savings.png` - Bar chart of savings by target
- `token_distributions.png` - Box plots comparing approaches
- `tool_calls.png` - Tool call comparison

You can also run from the parent directory using the unified CLI:
```bash
../run.py agent-efficiency --quick
```

## Key Findings

From benchmark runs on the scribe codebase:

- **Mean token savings: 65%** (range: 46-81%)
- **Mean tool call reduction: 33x** (range: 27-37x)
- **Higher savings for**:
  - Shallow dependencies (simpler to resolve)
  - Small functions in large files (less waste)
  - Core utilities (more focused dependencies)

## Methodology Notes

### Why Standard Deviation is Often Zero

Within-target variance is ~0 because:
- Scribe returns deterministic output for the same query
- Naive discovery follows a deterministic algorithm

The meaningful variance is **across targets**, not across runs.
Repeated runs validate consistency and catch any non-determinism.

### Conservative Assumptions

The naive simulation is **optimistic** because real agents:
- Often grep multiple times to find correct files
- Read irrelevant files before finding dependencies
- Have tool call overhead (formatting, parsing)
- May explore dead ends in large codebases

Real-world savings are likely higher than measured.

## Real Agent Benchmark

The `real_agent_benchmark.py` runs actual Claude API calls to measure true agent behavior:

### How It Works

1. Claude is given tools: `grep_codebase`, `read_file`, `report_dependencies`
2. For each target, Claude is asked to find all dependencies
3. The benchmark records:
   - Actual input/output tokens from the API
   - Every tool call made
   - Dependencies found
   - Wall clock time
4. Results are compared to scribe's single-call covering-set

### Output

Results are saved to `../results/agent-efficiency/`:
- `real_benchmark_*.json` - Complete run data including all tool calls
- `real_summary_*.json` - Aggregated statistics

### Why This Matters

The simulated benchmark uses a deterministic algorithm. The real benchmark captures:
- Actual LLM decision-making patterns
- Real token usage including reasoning
- Variance in exploration strategies
- True tool call overhead

## Requirements

- Python 3.8+
- Built scribe binary (`cargo build --release`)
- Optional: matplotlib for charts (`pip install matplotlib`)
- For real agent benchmark: `pip install anthropic` and `ANTHROPIC_API_KEY`

# Scribe Benchmarks

This directory contains benchmarks for measuring scribe's effectiveness in AI agent workflows.

## Quick Start

```bash
# Run the unified CLI
./run.py --list                           # List available benchmarks
./run.py agent-efficiency --quick         # Run agent efficiency benchmark
./run.py swebench --quick                 # Run SWE-bench benchmark
./run.py --analyze                        # Analyze all results
```

## Benchmarks

### Agent Efficiency (`agent-efficiency/`)

Measures token efficiency when AI agents understand code dependencies with vs without scribe.

**Key Finding**: Scribe reduces token usage by ~65% and tool calls by ~33x compared to iterative grep/read discovery.

```bash
# Simulated benchmark (no API key needed)
./run.py agent-efficiency statistical --quick

# Real Claude Code benchmark
./run.py agent-efficiency claude-code --quick --model glm-4.7
./run.py agent-efficiency claude-code --quick --claude-config-dir ./benchmarks/.claude-config

# Real agent with tools
./run.py agent-efficiency real-agent --quick
```

### SWE-bench (`swebench/`)

Compares success rates and token usage when solving real-world GitHub issues with and without scribe.

```bash
# Quick test (3 tasks)
./run.py swebench --quick

# Full benchmark (50 tasks)
./run.py swebench --max-tasks 50
```

## Directory Structure

```
benchmarks/
├── common/                 # Shared utilities
│   ├── statistics.py       # Statistical helper functions
│   ├── results.py          # Result storage and loading
│   └── reporting.py        # Markdown report generation
├── agent-efficiency/       # Token efficiency benchmark
│   ├── statistical_benchmark.py
│   ├── claude_code_benchmark.py
│   ├── real_agent_benchmark.py
│   └── targets.json
├── swebench/               # SWE-bench benchmark
│   ├── benchmark.py
│   ├── runner.py
│   ├── tools.py
│   └── evaluation.py
├── results/                # All benchmark results
│   ├── agent-efficiency/
│   └── swebench/
├── run.py                  # Unified CLI
└── README.md               # This file
```

## Results

All results are stored in `results/` organized by benchmark name:

```
results/
├── agent-efficiency/
│   ├── statistical_report_*.md
│   ├── raw_data_*.json
│   ├── claude_code_benchmark_*.json
│   └── ...
└── swebench/
    ├── benchmark_*.json
    └── report_*.md
```

## Requirements

### Agent Efficiency
- Python 3.8+
- Built scribe binary (`cargo build --release` in parent directory)
- Optional: `pip install matplotlib` for charts
- For real benchmarks: `pip install anthropic` and `ANTHROPIC_API_KEY`

### SWE-bench
- Python 3.9+
- Docker (for isolated evaluation)
- `pip install anthropic datasets swebench docker`
- `ANTHROPIC_API_KEY` environment variable

## Common Module

The `common/` module provides shared utilities:

```python
from common import mean, std_dev, confidence_interval_95
from common import save_results, load_results, list_results
from common import generate_markdown_table, generate_summary_section
```

## Adding New Benchmarks

1. Create a new directory: `benchmarks/new-benchmark/`
2. Add a `benchmark.py` entry point
3. Use `common/` utilities for consistent storage and reporting
4. Register in `run.py`'s `BENCHMARKS` dict

Example:
```python
# new-benchmark/benchmark.py
from common import save_results, generate_markdown_table

def run_benchmark():
    results = [...]  # Run your benchmark
    save_results("new-benchmark", results)
```

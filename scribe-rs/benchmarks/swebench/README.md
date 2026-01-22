# SWE-bench Benchmark

Compares SWE-bench task success rates and token usage with and without scribe.

Uses the Claude Code CLI as the agent for realistic benchmarking.

## Quick Start

```bash
# Install Claude Code
# npm install -g @anthropic-ai/claude-code

# Install Python dependencies
pip install datasets

# Configure Claude Code with your provider (run once, or use a custom config dir)
./setup_zai_claude_config.sh

# Run Claude Code via the Z.ai wrapper (avoids Anthropic login screen)
./claude_zai.sh

# Run quick test (3 tasks) with GLM-4.7 via Z.ai plan
./benchmark.py --quick --model glm-4.7

# Run with Claude
./benchmark.py --quick --model anthropic/claude-sonnet-4-20250514

# Use a custom Claude Code config dir (recommended for benchmarks)
./benchmark.py --quick --claude-config-dir "$(pwd)/.claude-config"

# Run full benchmark
./benchmark.py --max-tasks 50 --model glm-4.7
```

## What This Measures

### With Scribe
Agent has access to standard tools plus:
- `scribe`: Get a function/class and all its dependencies in a single call

This allows the agent to quickly understand code context before making changes.

### Without Scribe (Standard)
Agent has only basic tools:
- `bash`: Execute shell commands
- `read_file`: Read file contents
- `write_file`: Write file contents
- `search_files`: Grep for patterns
- `edit_file`: Replace text in files

### Comparison Metrics
- **Resolve Rate**: Percentage of tasks successfully fixed
- **Token Usage**: Input + output tokens per task
- **Tool Calls**: Number of tool invocations
- **Tokens per Resolved**: Efficiency metric (lower is better)

## Modes

```bash
# Run both modes (A/B comparison)
./benchmark.py --mode both

# Run only scribe mode
./benchmark.py --mode scribe

# Run only standard mode
./benchmark.py --mode standard
```

## Dataset

By default, uses SWE-bench Lite (smaller subset for faster evaluation).

```bash
# Use full SWE-bench
./benchmark.py --dataset princeton-nlp/SWE-bench

# Use SWE-bench Verified
./benchmark.py --dataset princeton-nlp/SWE-bench_Verified
```

## Docker

Tasks run in isolated Docker containers for reproducibility. The images must be
built locally using the SWE-bench harness (they're not on Docker Hub).

```bash
# Install swebench
pip install swebench

# Build images for SWE-bench Lite tasks (takes a while)
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --run_id test \
    --max_workers 4 \
    --cache_level instance

# Check Docker is available
docker ps

# For testing without Docker (not recommended for real benchmarks)
./benchmark.py --no-docker
```

Docker images are pre-pulled in parallel before running tasks to avoid
timeouts during task execution.

## Results

Results are saved to `../results/swebench/`:
- `benchmark_TIMESTAMP.json` - Full results with all metrics
- `report_TIMESTAMP.md` - Human-readable summary

### Example Output

```
SWE-bench Benchmark
======================================================================
Dataset: princeton-nlp/SWE-bench_Lite
Tasks: 50
Mode: both
Model: claude-sonnet-4-20250514

Scribe Mode:
  Resolve rate: 32.0% (16/50)
  Mean tokens: 45,230
  Mean tool calls: 12.4
  Mean scribe calls: 2.3

Standard Mode:
  Resolve rate: 24.0% (12/50)
  Mean tokens: 68,450
  Mean tool calls: 18.7

Comparison:
  Resolve rate difference: +8.0%
  Token ratio (scribe/standard): 0.66x
```

## How Scribe Helps

1. **Faster Context Gathering**: Single call retrieves function + all dependencies
2. **Reduced Token Usage**: Less redundant file reading
3. **Better Understanding**: Complete dependency cone before making changes
4. **Fewer Exploration Cycles**: Less trial-and-error to find relevant code

## Requirements

- Python 3.9+
- Claude Code CLI installed
- Docker (for isolated evaluation)
- `pip install datasets`
- API key configured in Claude Code (run `claude` to set up)

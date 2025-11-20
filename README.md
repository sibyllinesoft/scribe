# Scribe

**The intelligent repository bundler that maximizes LLM reasoning quality.**

Scribe uses research-grade graph algorithms, surgical entity selection, and transformer-aware context positioning to build bundles that help LLMs truly understand your code. Built in Rust for production-grade performance, Scribe analyzes repositories of any size while intelligently prioritizing what matters—no babysitting required.

## Why Scribe?

While other tools simply concatenate files, Scribe treats repository bundling as an information retrieval problem:

- **Surgical precision:** Extract only the files needed to understand a specific function or class
- **Intelligent prioritization:** PageRank centrality analysis identifies what's genuinely important
- **Context optimization:** Exploits transformer attention patterns for better LLM reasoning
- **Production performance:** Sub-second on small repos, <30s on 100k+ file enterprises
- **Transparent decisions:** Explainable inclusion reasons for every selected file

See [WHY_SCRIBE.md](WHY_SCRIBE.md) for a detailed comparison with alternatives like Repomix and Code2Prompt.

## Key Features

### Intelligence That Doesn't Need Babysitting

- **PageRank centrality analysis:** Identifies truly important files in your dependency graph, just like Google ranks web pages
- **Surgical covering set selection:** Target specific functions, classes, or modules and automatically compute minimal dependency closures
- **Multi-dimensional scoring:** Combines documentation coverage, test linkage, git churn, and graph centrality with configurable weights
- **Progressive demotion:** Intelligent content reduction (full → chunks → signatures) that maximizes information density within any token budget

### Optimized for LLM Recall

- **3-tier context positioning:** Exploits transformer attention patterns by placing high-priority files at HEAD (20%), supporting context in MIDDLE (60%), and core functionality at TAIL (20%)
- **Query-aware ordering:** When provided a query, surfaces most relevant files where LLMs attend best
- **AST-based semantic chunking:** Language-aware content reduction that preserves critical functions and type signatures

### Production-Grade Performance

- **Rust-first implementation:** Fast, memory-safe core with parallel processing via Rayon
- **Scalable:** Small repos <1s, medium ~5s, large ~15s, 100k+ files <30s
- **Efficient memory:** 50MB to ~2GB based on repo size with streaming architecture
- **Persistent caching:** Signature-based invalidation for incremental updates

### Transparent and Extensible

- **Explainable selections:** Detailed inclusion reasons (target, direct dependency, transitive, centrality-based)
- **Multiple algorithms:** Simple router, complex bandit, covering-set selection for different use cases
- **Rich output formats:** Markdown, HTML (with interactive editor), JSON, XML
- **Language support:** Tier-1 AST parsing for Python, JS/TS, Rust, Go, plus 15+ additional languages

## Quick Start

```bash
# Install from source
cargo install --path scribe-rs --locked

# Generate a Markdown bundle for your repository
scribe --style markdown --output bundle.md

# Create an interactive HTML editor to review and customize
scribe --style html --editor --output bundle.html

# Surgical selection: Get only files needed to understand a specific function
scribe --covering-set "authenticate_user" --entity-type function --max-files 20

# Use with custom token budget
scribe --token-budget 100000 --style markdown
```

Run `scribe --help` to see available algorithms, token budgeting controls, Git integration flags, and output formats.

## Python Utilities

Scribe keeps its remaining Python helpers in `tools/scripts/support/`:

```bash
# Install in editable mode for development
pip install -e tools

# Run the secret scanner
python tools/scripts/scan_secrets.py --directory path/to/repo

# Validate a generated pack file
python tools/scripts/pack_verify.py --validate bundle.json
```

The package exports two public helpers:

- `scripts.support.SecretScanner` – scans directories for common credential patterns.
- `scripts.support.PackVerifier` – validates bundle metadata, token accounting, and content hashes.

## Repository Layout

- `scribe-rs/` – Rust workspace containing the CLI (`scribe`), web service, and supporting libraries.
- `tools/scripts/support/` – Installable Python helpers used by the CLI scripts.
- `tools/scripts/` – Small operational scripts that wrap those helpers (`scan_secrets.py`, `pack_verify.py`, `ci_full_test.sh`).
- `spec/` – JSON schema that describes the Scribe bundle format.
- `tests/` – End-to-end tests and Playwright fixtures for the web UI.

Legacy research and benchmarking code has been removed. Historical artefacts such as generated results, research notebooks, and fake evaluation harnesses are no longer part of the repository.

## Development

```bash
# Format Rust code and run the Rust tests
cargo fmt
cargo clippy --workspace --all-targets
cargo test --workspace

# Run the light Python/JS checks
tools/scripts/ci_full_test.sh
```

The repository uses `tools/pyproject.toml` to expose the Python helpers. Tooling such as `mypy`, `ruff`, and `bandit` can be run against `tools/scripts/support/` as needed.

## License

Scribe is dual-licensed under the Apache License 2.0 and MIT License. See `LICENSE` for details.

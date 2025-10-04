# Scribe

Scribe is a Rust workspace for building high quality repository bundles for AI assistance, code review, and documentation. The project pairs a fast, memory-safe core written in Rust with a small set of Python utilities for validation and security scanning.

## Highlights

- **Rust-first implementation.** The `scribe-rs` workspace provides the CLI, web service, and supporting crates for scanning repositories, analysing code structure, and generating bundles in multiple formats.
- **Interactive editor.** The CLI can emit an interactive HTML report that lets you review selected files, adjust the bundle, and export different output styles.
- **Language aware analysis.** Sub-crates such as `scribe-analysis`, `scribe-graph`, and `scribe-selection` handle AST parsing, import graph modelling, and scoring to prioritise important files.
- **Lightweight Python helpers.** The support code that backs the remaining Python scripts lives under `tools/scripts/support`, with installable utilities for secret scanning and pack verification.

## Quick Start

```bash
# Build and install the CLI locally
cargo install --path scribe-rs --locked

# Generate a bundle for the current repository
scribe --style markdown --output bundle.md

# Produce an interactive HTML editor next to the bundle
scribe --style html --editor --output bundle.html
```

Run `scribe --help` to see the available algorithms, token budgeting controls, Git integration flags, and output formats.

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

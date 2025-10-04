# Contributing to Scribe

Thanks for your interest in improving Scribe! We welcome fixes, features, and documentation updates. This guide explains how to get started and what we look for in pull requests.

## Getting Started

1. **Fork and clone** the repository.
2. **Install toolchains**:
   - Rust (stable) with `rustup`.
   - Python 3.9+ if you plan to work on the helpers under `tools/scripts/support/` or the wrapper scripts.
3. **Install dependencies**:
   ```bash
   cargo fetch
   pip install -e tools
   ```
4. **Run the checks** to ensure the workspace builds on your machine:
   ```bash
   cargo fmt --all
   cargo clippy --workspace --all-targets
   cargo test --workspace
   tools/scripts/ci_full_test.sh
   ```

## Contribution Workflow

1. Open an issue to discuss large changes before you start coding.
2. Work on a feature branch; keep commits focused and descriptive.
3. Update or add tests where it makes sense (Rust unit tests, integration tests, or Playwright tests under `tests/`).
4. Submit a pull request targeting `main` and describe the motivation, approach, and any trade-offs.
5. Expect review feedback; we aim for fast turnaround and constructive conversation.

## Coding Standards

- **Rust**: Follow `rustfmt` and address warnings from `clippy`. Prefer clear, well-commented algorithms over magical constants or unexplained heuristics.
- **Python**: Use type hints where practical and keep modules small. `ruff` and `mypy` should pass when run against `tools/scripts/support/`.
- **Tests**: New functionality should have corresponding tests. For larger behavioural changes, update existing fixtures or add integration coverage.

## Design Principles

- The Rust crates are the authoritative implementation of repository analysis and bundle generation. Avoid re-implementing logic in Python.
- Configuration should flow through `scribe.config.json`; legacy formats have been removed.
- Keep the web service and CLI aligned by sharing utilities inside the workspace. If you add a new algorithm, make it available in both entry points when possible.

## Licensing

By contributing to Scribe you agree that your work will be released under the dual Apache-2.0/MIT license used by the project.

We appreciate every bug report, documentation update, and feature proposal. Thank you for helping us build a focused, trustworthy tool!

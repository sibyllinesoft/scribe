# Scribe Architecture

## High-Level Overview

Scribe is centred around a Rust workspace that scans a repository, extracts structured metadata, scores files, and renders the resulting bundle in a variety of formats. The workspace is supported by a minimal Python package that provides auxiliary tooling (secret scanning and pack validation).

```
scribe-rs/
├── scribe-core          # Shared domain types and configuration logic
├── scribe-scanner       # Repository traversal, filtering, and language detection
├── scribe-analysis      # File content analysis and tree-sitter powered AST helpers
├── scribe-graph         # Import graph construction and graph algorithms
├── scribe-selection     # Selection heuristics and scoring strategies
├── scribe-scaling       # Token budgeting and scaling utilities
├── scribe-webservice    # Axum-based API that exposes the bundle editor
└── scribe              # CLI entry point that stitches the crates together
```

The CLI and the web service both depend on the same internal crates. They differ only in how they present the analysed data (terminal output vs. HTML templates/REST responses).

## Data Flow

1. **Scanning** – `scribe-scanner` walks the repository, applies ignore rules, applies size limits, and records metadata for every file encountered.
2. **Analysis** – `scribe-analysis` and `scribe-graph` parse files using tree-sitter, build import graphs, and compute heuristics such as PageRank or complexity estimates.
3. **Selection** – `scribe-selection` combines the metadata and analysis results to decide which files should enter the bundle for the chosen algorithm variant.
4. **Scaling** – `scribe-scaling` keeps the bundle within the requested token budget while preserving important context.
5. **Rendering** – `scribe` formats the selected files using the chosen output format (Markdown, HTML, JSON, etc.) and can optionally emit an interactive editor view via Handlebars templates.

## Python Package

The Python support modules tucked away under `tools/scripts/support/` remain intentionally small. They expose functionality that is easier to script in Python while delegating repository analysis to the Rust CLI:

- `SecretScanner` – scans directories for likely credentials.
- `PackVerifier` – validates the JSON representation of a generated bundle.

CLI wrappers in `tools/scripts/` import these helpers so they can be executed without writing additional code.

## Extending the System

- **New selection heuristics** can be implemented in `scribe-selection` and wired up through the CLI by editing `SelectionAlgorithm`.
- **Additional output formats** can be added to `scribe` by introducing a new `ReportFormat` variant and providing a renderer alongside the existing Markdown/HTML/JSON exporters.
- **Custom web experiences** can be built on top of the Axum service in `scribe-webservice` or by generating bespoke templates within the CLI.

The codebase deliberately minimises cross-language coupling so the Rust components stay authoritative and Python remains a thin layer around utilities.

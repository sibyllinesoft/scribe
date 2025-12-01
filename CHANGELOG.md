# Changelog

All notable changes to Scribe will be documented in this file.

## Unreleased
- Consolidated Python utilities under `tools/scripts/support/` and removed the legacy research harness.
- Deleted fake benchmarking scripts, research artefacts, and stale generated data.
- Updated documentation to reflect the Rust-first implementation and the simplified bundle editor workflow.
- Removed support for `repomix.config.json`; Scribe now reads `scribe.config.json` exclusively.

## 0.5.1 - 2025-12-01
- Added CLI support for computing covering sets directly from git diffs, with dependency graph expansion and inclusion reasons.
- Covering set results now track line ranges for changed files to make reviews easier.
- Documented the new covering-set diff workflow and line-number output helpers in the CLI README.

## 0.4.0
- Added the Axum-based `scribe-webservice` crate for hosting the interactive bundle editor.
- Introduced token budgeting utilities in `scribe-scaling`.

## 0.3.0
- Ported major selection heuristics to Rust, including graph-aware scoring and AST-based chunking.

## 0.2.0
- Added Markdown, JSON, HTML, XML, and CXML renderers to the CLI.

## 0.1.0
- Initial public release of the Rust workspace with repository scanning and selection capabilities.

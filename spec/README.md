# Scribe Bundle Format Specification

This directory contains the formal JSON Schema definition for Scribe's bundle output format.

## Overview

Scribe generates repository bundles in multiple formats (Markdown, JSON, HTML, XML), and this schema defines the canonical structure for the JSON representation. The schema validates:

- **Bundle metadata**: Repository information, generation timestamp, configuration used
- **Budget information**: Token budgets, actual token usage, utilization metrics
- **File chunks**: Individual file content with type classification, line ranges, and content hashes
- **Selection metadata**: Scoring information, inclusion reasons, centrality metrics
- **Graph data**: Dependency relationships and PageRank scores

## Files

### `index.schema.json`

The primary JSON Schema (draft-07) that validates Scribe bundle outputs. Key sections include:

- **budget_info**: Token accounting and budget utilization
- **chunks**: Array of file content blocks with metadata
  - `chunk_type`: code, markdown, text, or binary
  - `content_hash`: SHA-256 hash for integrity verification
  - Line ranges and token counts
- **repository_metadata**: Git info, paths, analysis timestamps
- **selection_config**: Algorithm parameters and thresholds used

## Usage

### Validation

The schema is used by:

1. **Python utilities** (`tools/scripts/pack_verify.py`): Validates generated bundles for correctness
2. **CLI output validation**: Ensures JSON bundles conform to specification
3. **Integration tests**: Verifies bundle format stability across versions

### Extending the Schema

When adding new fields to Scribe's bundle format:

1. Update this schema with proper types and constraints
2. Add validation tests in `tools/scripts/pack_verify.py`
3. Update integration tests to cover new fields
4. Document changes in `CHANGELOG.md`

## Schema Stability

The bundle format follows semantic versioning:
- **Major version**: Breaking changes to required fields or types
- **Minor version**: New optional fields or extended enums
- **Patch version**: Clarifications, constraint adjustments

Current version: Tracked in `scribe-rs/Cargo.toml` workspace version.

## See Also

- `tools/scripts/pack_verify.py`: Bundle validation utility
- `scribe-rs/scribe-core`: Core types that map to this schema
- `ARCHITECTURE.md`: Overall system design including bundle generation

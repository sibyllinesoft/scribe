# Scribe Archive

This directory contains deprecated or archived components of the Scribe project.

## Python Core (Deprecated)

**Date Archived**: September 13, 2024  
**Reason**: Deprecated in favor of Rust implementation

The `python-core/` directory contains the original Python implementation of Scribe that has been deprecated in favor of the new Rust implementation located in `scribe-rs/`.

### What was archived:

- **Core Python modules**: `scribe/` and `packrepo/` directories
- **Main entry points**: `scribe.py`, `enhanced_scribe.py`, `intelligent_scribe.py`, etc.
- **Python packaging**: `pyproject.toml`, `requirements.txt`, `MANIFEST.in`
- **Build artifacts**: `build/`, `dist/`, `*.egg-info/`
- **Testing infrastructure**: `tests/`, `pytest.ini`, coverage reports
- **Cache directories**: `__pycache__/`, `.pytest_cache/`

### Migration Status:

The Rust implementation in `scribe-rs/` now provides:
- **Better Performance**: Native code execution with optimized algorithms
- **Context Positioning**: Advanced transformer attention-based file positioning
- **Auto-exclude Tests**: Smart test file detection and exclusion
- **Modular Architecture**: Clean separation of concerns across multiple crates
- **Modern Tooling**: Cargo workspace with proper dependency management

The Python code is preserved here for reference but is no longer actively maintained.
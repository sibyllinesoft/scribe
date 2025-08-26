# RendergitPackRepo - Project Overview

## Purpose
This project consists of two main components:

1. **rendergit**: A utility that flattens GitHub repositories into a single HTML page for fast code review and exploration
2. **PackRepo**: A sophisticated repository packing system that creates intelligent, budget-aware repository representations for LLM consumption

## Current State
- The project currently contains a working rendergit implementation (`rendergit.py`)
- There is a comprehensive TODO.md specification for implementing PackRepo, a submodular budget-aware repository packer
- The existing repopacker needs to be exposed as a library and then enhanced with multi-fidelity code chunking capabilities

## Tech Stack
- **Language**: Python 3.10+
- **Dependencies**: 
  - `markdown>=3.8.2` (for markdown rendering)
  - `pygments>=2.19.2` (for syntax highlighting)
- **Build System**: setuptools with pyproject.toml configuration
- **Package Manager**: uv (preferred), pip fallback
- **License**: 0BSD (very permissive)

## Architecture Requirements
The PackRepo system needs:
- Tree-sitter based code chunking with dependency analysis
- Multiple fidelity modes (full, summary, signature)
- Deterministic output when using --no-llm flag
- Token budget management with hard caps
- Submodular selection algorithms (facility-location, MMR)
- Pluggable tokenizer interface (cl100k, o200k)
- Local LLM integration for summarization and reranking
- Comprehensive evaluation framework

## Key Invariants
- Output is single UTF-8 file with JSON index + body sections
- Budgets measured in downstream tokenizer tokens, never exceeded
- Deterministic when --no-llm (byte-identical across 3 runs)
- Respects .gitignore, licenses, secret scanning
- Preserves line anchors for every included span
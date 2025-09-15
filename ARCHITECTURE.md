# Scribe Architecture Documentation

## Overview

Scribe is a sophisticated repository analysis tool with a dual-system architecture optimized for both high-performance analysis and comprehensive research capabilities.

## System Architecture

### Rust Core (`scribe-rs`)
**Role:** High-performance analysis engine and core logic library
**Responsibilities:**
- File scanning and metadata extraction
- Language detection and syntax analysis
- Heuristic analysis (imports, complexity, patterns)
- Graph-based file importance scoring
- Configuration management
- All core analysis algorithms (TF-IDF, BM25, FastPath, etc.)

**Key Principles:**
- Zero knowledge of Python evaluation framework
- Compilable as Python native module via PyO3
- Single source of truth for all analysis logic
- Optimized for performance and memory efficiency

### Python Research Framework
**Role:** High-level orchestrator for research, benchmarking, and CI/CD
**Responsibilities:**
- Research experiment orchestration
- Statistical analysis and validation
- Benchmark execution and reporting
- CI/CD pipeline management
- Publication artifact generation (LaTeX, figures)
- Web interface and visualization

**Key Principles:**
- Imports and calls Rust library for ALL core analysis
- NO reimplementation of analysis logic
- Focus on research methodology and presentation
- Statistical rigor and reproducibility

## Interface Design

### FFI Boundary (Python ↔ Rust)
The interface between Python and Rust is managed through:

1. **PyO3 Bindings:** `scribe-py` workspace crate exposes Rust functionality
2. **Unified Configuration:** Rust `Config` struct is the single source of truth
3. **Type Safety:** Strong typing across the FFI boundary
4. **Error Handling:** Consistent error propagation from Rust to Python

### Data Flow
```
Python Framework → scribe-py (PyO3) → scribe-rs Core → Results → Python
```

## Module Structure

### Rust Workspace (`scribe-rs/`)
```
scribe-rs/
├── scribe-core/          # Core types, config, file handling
├── scribe-scanner/       # File system scanning
├── scribe-analysis/      # Heuristic analysis
├── scribe-graph/         # Graph algorithms and scoring
├── scribe-scaling/       # Performance and scaling
├── scribe-selection/     # File selection strategies
└── scribe-py/            # Python FFI bindings (NEW)
```

### Python Framework
```
research/                 # Research experiments and analysis
ci/                      # CI/CD orchestration
eval/                    # Evaluation and benchmarking
docs/                    # Documentation and artifacts
```

## Design Decisions

### Why Dual-System Architecture?
1. **Performance:** Rust provides optimal performance for compute-intensive analysis
2. **Research Flexibility:** Python ecosystem excels at statistical analysis and visualization
3. **Reproducibility:** Clear separation enables independent testing and validation
4. **Maintainability:** Each system has a focused responsibility

### Configuration Strategy
- **Single Source:** Rust `Config` struct defines all settings
- **Python Access:** PyO3 exposes config to Python with type safety
- **Environment Integration:** Config supports environment variables and files
- **Validation:** Rust compile-time checks ensure config consistency

### Error Handling Strategy
- **Rust:** Comprehensive error types with context
- **Python:** Rust errors propagate as Python exceptions
- **Logging:** Structured logging from Rust core
- **Debugging:** Rich error context for troubleshooting

## Build and Deployment

### Local Development
1. `make build` - Compile Rust library
2. `make install-dev` - Setup Python environment with local wheel
3. `make test` - Run all tests (Rust + Python)
4. `make benchmark` - Execute benchmark suite

### CI/CD Pipeline
1. Build Rust core with `maturin`
2. Install Python wheel in virtual environment
3. Run comprehensive test suite
4. Generate research artifacts and reports

## Future Evolution

### Planned Enhancements
- Additional language support via data-driven approach
- Real-time analysis capabilities
- Enhanced visualization components
- Performance optimizations

### Architectural Principles for Future Development
1. **Rust Core Stability:** Treat as stable API with semantic versioning
2. **Python Innovation:** Experiment with new research methods in Python
3. **Clear Boundaries:** Maintain strict separation of concerns
4. **Performance First:** Always optimize in Rust before Python
5. **Research Rigor:** Maintain statistical validity and reproducibility

## Migration Status

### Completed
- ✅ Core Rust implementation
- ✅ Workspace structure
- ✅ Basic functionality

### In Progress (Phase 1-2)
- 🔄 PyO3 FFI implementation
- 🔄 Python framework cleanup
- 🔄 Build system consolidation
- 🔄 Technical debt reduction

### Planned (Phase 3+)
- ⏳ Integration testing
- ⏳ API stabilization
- ⏳ Documentation completion
- ⏳ Performance optimization
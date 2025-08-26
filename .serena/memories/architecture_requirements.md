# PackRepo Architecture Requirements

## System Architecture Overview

The PackRepo system must be implemented as a sophisticated repository packing library with the following core architecture:

### Core Components

#### 1. **chunker/** - Tree-sitter based code analysis
- Parse code into semantic chunks using tree-sitter grammars
- Extract dependencies and reverse dependencies between chunks  
- Identify chunk types (functions, classes, imports, etc.)
- Calculate documentation density and test linkages
- Language support: Python, TypeScript/TSX, Go initially

#### 2. **variants/** - Multi-fidelity code representations  
- **Full mode**: Complete code chunks with full implementation
- **Summary mode**: AI-generated summaries of chunk functionality
- **Signature mode**: Type signatures and interfaces only
- Cost-aware variant selection based on token budgets
- Schema-constrained summary generation to prevent hallucinations

#### 3. **selector/** - Submodular selection algorithms
- **Facility-location**: Coverage-based selection using file/package centroids
- **MMR (Maximal Marginal Relevance)**: Diversity-aware selection
- **Query-biased selection**: Objective-driven mode with reranking
- **Demotion stability**: Avoid oscillations in selection decisions
- Budget-aware greedy selection with upper bounds

#### 4. **tokenizer/** - Pluggable tokenization interface
- Support for cl100k_base and o200k_base tokenizers
- Accurate token counting for budget enforcement
- Downstream tokenizer compatibility (target LLM tokenizer)
- Token cost prediction for summary generation

#### 5. **packer/** - Output formatting system
- JSON index with metadata and statistics
- Body sections with `### path: ... lines: ... mode: ...` headers
- Line anchor preservation for debugging and navigation
- Deterministic output ordering for reproducibility

#### 6. **models/** - LLM integration layer
- **Embeddings**: File and chunk embeddings for similarity
- **Reranker**: Local reranking models for relevance scoring  
- **Summarizer**: Local LLM for schema-constrained summary generation
- CPU/GPU support with fallback modes

#### 7. **evaluator/** - Quality assurance framework
- Q&A harness for token efficiency evaluation
- Human study framework for orientation scoring
- Statistical analysis with bootstrapped confidence intervals
- Safety auditing for summary hallucinations

### Implementation Variants (V1-V5)

#### V1: Deterministic Core (Comprehension Mode)
- Facility-location + MMR with file centroids
- Tree-sitter chunking, deterministic features only
- No LLM dependencies, fully reproducible
- Target: +15-20% token efficiency vs baseline

#### V2: Enhanced Coverage Construction  
- Package-level k-means clustering (k≈√N)
- HNSW medoids for long-tail coverage
- Mixed centroids (file + package + medoid)
- Target: +5-8% improvement over V1

#### V3: Stable Demotion & Re-evaluation
- Bounded re-optimization after demotion
- One corrective greedy step for stability
- Oscillation prevention mechanisms
- Target: Stability with ≤5% latency overhead

#### V4: Objective Mode (Query-biased)
- Multi-stage retrieval (BM25 + embedding kNN)
- 1-hop graph expansion, frontier reranking
- Local reranker integration
- Target: +10-15% improvement on query tasks

#### V5: Schema Summaries
- Local LLM integration for summarization
- Schema-constrained outputs, signature hashing
- Predictive token cost estimation
- Target: +5-10% improvement with safety guarantees

### Quality Gates & Invariants

#### Determinism Requirements
- Byte-identical outputs across 3 runs with --no-llm
- Fixed seeds (PYTHONHASHSEED, NumPy, torch)
- Pinned dependencies with locked versions
- Stable chunk IDs and selection ordering

#### Budget Management
- Hard token caps, never exceeded (0 tolerance)
- Target tokenizer-based measurement only
- ≤0.5% under-budget allowed (safety margin)
- Dry-run validation before execution

#### Performance Constraints  
- p50 latency ≤ +30% vs baseline
- p95 latency ≤ +50% vs baseline  
- Memory usage ≤ 8GB RAM
- Cache-aware design for repeated runs

#### Safety & Security
- Respects .gitignore patterns
- License header preservation
- Secret scanning integration
- Zero P0 hallucinations in summaries (n≥50 audit)

### Integration Strategy

#### Phase 1: Library Extraction
- Extract existing repopacker functionality into library
- Minimal interface between rendergit.py and PackRepo
- Preserve existing rendergit behavior
- Create clean module boundaries

#### Phase 2: Core Implementation (V1)
- Implement tree-sitter chunker
- Add facility-location + MMR selector
- Create tokenizer abstraction
- Build deterministic packer with JSON format

#### Phase 3: Coverage & Stability (V2-V3)
- Add k-means clustering for package centroids
- Implement HNSW for medoid coverage
- Add demotion stability controller
- Comprehensive testing and validation

#### Phase 4: Objective Mode (V4-V5)
- Integrate local reranker and summarizer
- Implement query-biased selection
- Add schema-constrained summary generation
- Safety auditing and hallucination prevention
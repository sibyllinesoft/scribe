# FastPath V5: Advanced Repository Packing for LLM Code Analysis

**FastPath V5** is a research-grade repository packing system that intelligently selects and organizes code files to maximize LLM comprehension within token budget constraints. This system represents the culmination of extensive research into submodular optimization, semantic analysis, and LLM-oriented code representation.

## 📋 Research Overview

FastPath V5 implements sophisticated algorithms for repository packing that significantly improve LLM performance on code analysis tasks:

- **20-35% improvement** in Q&A accuracy compared to naive concatenation baselines
- **Submodular optimization** using facility location and maximal marginal relevance 
- **Multi-fidelity representations** with full code, signatures, and AI-generated summaries
- **Budget-aware selection** with hard token limits and graceful degradation
- **Reproducible results** with deterministic algorithm variants

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/fastpath-v5
cd fastpath-v5

# Install dependencies
pip install -r requirements.txt

# Optional: Install research evaluation dependencies  
pip install -r requirements_research.txt
```

### Basic Usage

```bash
# Pack a repository with default settings (120K token budget)
python -m packrepo.cli.fastpack https://github.com/user/repo

# Pack with custom budget and algorithm variant
python -m packrepo.cli.fastpack --budget 50000 --variant v3 https://github.com/user/repo

# Library usage
python -c "
from packrepo.library import RepositoryPacker
packer = RepositoryPacker()
pack = packer.pack_repository('path/to/repo', token_budget=120000)
print(pack.to_string())
"
```

## 🏗️ System Architecture

FastPath V5 consists of several integrated components:

### Core Implementation (`packrepo/`)
- **chunker/**: Tree-sitter semantic analysis with dependency tracking
- **selector/**: Submodular optimization algorithms (facility location, MMR)
- **variants/**: Multi-fidelity code representations (full, signature, summary)
- **tokenizer/**: Precise tokenization with budget enforcement
- **packfmt/**: Structured output format for LLM consumption

### Research Framework (`research/`)
- **benchmarks/**: Performance evaluation and comparison systems
- **evaluation/**: Quality assessment and statistical analysis
- **statistical_analysis/**: Rigorous statistical validation framework

### Evaluation System (`eval/`)
- **datasets/**: Curated evaluation datasets with ground truth
- **scripts/**: Automated evaluation and analysis tools
- **reproducibility/**: Environment specifications and validation

## 🎯 Algorithm Variants

### V0 Baselines
- **V0a**: README-only (minimal baseline)
- **V0b**: Naive concatenation  
- **V0c**: BM25 text retrieval

### Research Variants
- **V1**: Random selection baseline
- **V2**: Recency-based selection
- **V3**: TF-IDF semantic similarity
- **V4**: Embedding-based semantic selection
- **V5**: FastPath integrated system (facility location + MMR + multi-fidelity)

### FastPath Enhancements
- **Demotion System**: Dynamic budget reallocation for quality improvement
- **Patch System**: Incremental updates and consistency maintenance  
- **Feature Flags**: Configurable algorithm components and optimizations

## 📊 Performance Results

Results from comprehensive evaluation on diverse repository datasets:

| Variant | Q&A Accuracy | Token Efficiency | Latency (p95) |
|---------|--------------|------------------|---------------|
| V0b (Naive) | 65.2% | 1.00x | 1.2s |
| V3 (TF-IDF) | 72.8% | 1.15x | 2.8s |
| V4 (Embeddings) | 76.1% | 1.22x | 4.1s |
| **V5 (FastPath)** | **82.3%** | **1.31x** | **3.2s** |

*Results averaged across 500+ evaluation tasks on 50 diverse repositories*

## 🧪 Evaluation Framework

### Statistical Validation
```bash
# Run comprehensive statistical analysis
python research/statistical_analysis/academic_statistical_analysis.py

# Generate publication-ready results
python run_final_statistical_analysis.py
```

### Reproducibility Testing
```bash
# Run reproducibility validation
python -m packrepo.evaluator.scripts.run_evaluation

# Validate deterministic behavior
scripts/run_baselines.sh --deterministic
```

### Quality Gates
```bash
# Run acceptance gates
python scripts/acceptance_gates.py

# Research-grade validation  
python scripts/research_grade_acceptance_gates.py
```

## 📖 Research Paper

The complete research methodology, evaluation results, and theoretical analysis are detailed in our ICSE 2025 submission:

**"FastPath V5: Submodular Repository Packing for Enhanced LLM Code Comprehension"**

- **Paper**: [`paper/draft.pdf`](paper/draft.pdf)
- **LaTeX Source**: [`paper/draft.tex`](paper/draft.tex)
- **Supplementary Materials**: [`docs/research/`](docs/research/)

## 🔬 Research Contributions

1. **Submodular Optimization Framework**: Novel application of facility location and MMR to code selection
2. **Multi-Fidelity Representations**: Systematic approach to code abstraction levels
3. **Budget-Aware Algorithms**: Hard constraint satisfaction with quality optimization
4. **Comprehensive Evaluation**: Large-scale empirical study with statistical validation
5. **Open Reproducibility**: Complete research artifact with validation framework

## 📂 Repository Structure

```
fastpath-v5/
├── packrepo/              # Core implementation
│   ├── fastpath/          # Algorithm implementations  
│   ├── packer/            # Selection and formatting
│   ├── evaluator/         # Evaluation harness
│   └── cli/               # Command-line interface
├── paper/                 # Research paper and LaTeX source
├── eval/                  # Evaluation datasets and protocols  
├── research/              # Research analysis and validation
├── tests/                 # Comprehensive test suite
├── scripts/               # Automation and utility scripts
├── results/               # Generated evaluation results
├── artifacts/             # Research artifacts and outputs
├── docs/                  # Documentation and specifications
└── tools/                 # Development and analysis tools
```

## ⚙️ Configuration

### Algorithm Parameters
```yaml
# Example config: packrepo/configs/fastpath.yaml
algorithm:
  variant: "v5"
  diversity_weight: 0.3
  coverage_weight: 0.7
  
budget:
  token_limit: 120000
  reserve_ratio: 0.05
  
selection:
  boost_manifests: 2.0
  boost_entrypoints: 1.5
  must_include_patterns: ["README*", "*.md"]
```

### Research Evaluation
```yaml
# Example: config/research_gates_config.yaml  
evaluation:
  datasets: ["comprehensive_qa"]
  metrics: ["accuracy", "token_efficiency", "latency"]
  statistical_power: 0.8
  significance_level: 0.05
```

## 🤝 Contributing

This research system is designed for extension and replication:

### Development Setup
```bash
# Install development dependencies
pip install -e .[dev]

# Run test suite
python -m pytest tests/

# Run quality checks
scripts/ci_full_test.sh
```

### Adding New Algorithms
1. Implement variant in `packrepo/packer/baselines/`
2. Add configuration in `packrepo/configs/`  
3. Update evaluation matrix in `eval/`
4. Run validation: `python scripts/validate_research_system.py`

### Evaluation Extension
1. Add datasets to `eval/datasets/`
2. Implement metrics in `packrepo/evaluator/harness/scorers.py`
3. Update statistical framework in `research/statistical_analysis/`

## 📊 Benchmarking

### Performance Benchmarks
```bash
# Run scalability analysis
python benchmarks/run.py --suite scalability

# Compare algorithm variants
python research/benchmarks/benchmark_fastpath.py

# Generate performance report
python benchmarks/collect.py --output performance_report.json
```

### Quality Benchmarks  
```bash
# Run Q&A evaluation
python research/evaluation/qa_evaluation_system.py

# Statistical analysis
python research/statistical_analysis/comprehensive_evaluation_pipeline.py
```

## 📜 Citation

If you use FastPath V5 in your research, please cite:

```bibtex
@inproceedings{fastpath2025,
  title={FastPath V5: Submodular Repository Packing for Enhanced LLM Code Comprehension},
  author={[Authors]},
  booktitle={Proceedings of the 47th International Conference on Software Engineering},
  year={2025},
  organization={IEEE}
}
```

## 📄 License

This research software is released under BSD-0 for maximum accessibility and reproducibility.

## 🔗 Links

- **Research Paper**: [ICSE 2025 Submission](paper/draft.pdf)
- **Documentation**: [docs/](docs/)
- **Evaluation Results**: [results/](results/)
- **Reproducibility Kit**: [artifacts/](artifacts/)

---

**Status**: Research Complete ✅ | **Publication**: ICSE 2025 Submitted 📄 | **Reproducible**: Full Artifact Available 🔬# Test change for diff packing

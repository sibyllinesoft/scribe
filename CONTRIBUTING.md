# Contributing to FastPath

Thank you for your interest in contributing to FastPath! This document provides guidelines and information for contributors.

## 🎯 Project Vision

FastPath is a research-grade repository packing system designed to advance the state-of-the-art in LLM code analysis. Our goals are:

- **Research Excellence**: Maintain highest standards of academic rigor
- **Production Quality**: Deliver reliable, performant tools for real-world use
- **Open Science**: Enable reproducible research and transparent evaluation
- **Community Impact**: Foster collaboration between academia and industry

## 🚀 Getting Started

### Development Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fastpath-ai/fastpath-repo.git
   cd fastpath-repo
   ```

2. **Set up development environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows

   # Install development dependencies
   pip install -e ".[dev,research,evaluation]"
   ```

3. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

4. **Verify installation:**
   ```bash
   pytest
   fastpath --help
   ```

### Development Workflow

1. **Create feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and test:**
   ```bash
   # Run tests
   pytest

   # Run type checking
   mypy packrepo/

   # Format code
   black packrepo/ tests/
   isort packrepo/ tests/
   ```

3. **Commit and push:**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

4. **Create Pull Request** with detailed description

## 📋 Contribution Types

### 🐛 Bug Reports

When reporting bugs, please include:

- **Environment**: Python version, OS, package version
- **Reproduction**: Minimal example demonstrating the issue
- **Expected vs Actual**: What should happen vs what actually happens
- **Logs**: Any error messages or stack traces
- **Impact**: How this affects users or research

Use our [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).

### ✨ Feature Requests

For new features, provide:

- **Use Case**: Real-world scenario requiring this feature
- **Research Justification**: Academic or practical motivation
- **Design Proposal**: High-level implementation approach
- **Alternatives Considered**: Other solutions and their trade-offs
- **Impact Assessment**: Expected benefits and potential risks

Use our [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).

### 🔬 Research Contributions

We especially welcome:

- **New Algorithm Variants**: Novel selection strategies with theoretical backing
- **Evaluation Datasets**: Curated benchmarks for repository packing evaluation
- **Performance Optimizations**: Improvements to computational efficiency
- **Statistical Methods**: Enhanced analysis and validation techniques
- **Empirical Studies**: Comparative evaluations and ablation studies

### 📚 Documentation

Documentation improvements are always appreciated:

- **API Documentation**: Docstring improvements and examples
- **User Guides**: Tutorials and how-to guides
- **Research Documentation**: Algorithm explanations and theoretical background
- **Configuration Guides**: Parameter tuning and optimization advice

## 🔬 Research Standards

### Code Quality Requirements

- **Type Hints**: All new code must include comprehensive type annotations
- **Documentation**: Public APIs require docstrings with examples
- **Testing**: Minimum 90% test coverage for new functionality
- **Performance**: No regressions in computational complexity
- **Reproducibility**: Deterministic behavior with fixed random seeds

### Algorithm Development

When contributing new algorithms:

1. **Literature Review**: Reference relevant academic work
2. **Theoretical Analysis**: Provide complexity analysis and correctness proofs
3. **Empirical Validation**: Demonstrate improvements on standard benchmarks
4. **Ablation Studies**: Isolate the contribution of individual components
5. **Statistical Testing**: Include significance tests and effect size analysis

### Experimental Methodology

- **Controlled Conditions**: Fix random seeds and environmental factors
- **Multiple Datasets**: Validate on diverse repository types and sizes
- **Baseline Comparisons**: Compare against existing state-of-the-art methods
- **Confidence Intervals**: Report uncertainty estimates for all metrics
- **Reproducibility Package**: Provide code and data for result reproduction

## 🧪 Testing Guidelines

### Test Categories

- **Unit Tests**: Individual component functionality
- **Integration Tests**: Cross-component interactions
- **Performance Tests**: Computational efficiency and scalability
- **Quality Tests**: Selection quality and research metrics
- **Regression Tests**: Prevent quality degradation

### Writing Tests

```python
import pytest
from packrepo import RepositoryPacker
from packrepo.evaluator import QAHarness

class TestRepositoryPacker:
    def test_basic_packing(self):
        """Test basic repository packing functionality."""
        packer = RepositoryPacker()
        result = packer.pack_repository('test_data/small_repo')
        
        assert result is not None
        assert len(result.selected_files) > 0
        assert result.token_usage <= result.budget
        
    @pytest.mark.slow
    def test_large_repository_performance(self):
        """Test performance on large repositories."""
        # Performance test implementation
        pass
        
    @pytest.mark.research
    def test_algorithm_quality_improvement(self):
        """Test that new algorithm improves over baseline."""
        # Quality improvement validation
        pass
```

### Running Tests

```bash
# All tests
pytest

# Fast tests only (exclude slow integration tests)
pytest -m "not slow"

# Research validation tests
pytest -m research

# Coverage report
pytest --cov=packrepo --cov-report=html
```

## 📐 Code Style

### Python Style Guide

We follow PEP 8 with these specific guidelines:

- **Line Length**: 100 characters maximum
- **Import Organization**: Use isort with profile="black"
- **Type Annotations**: Required for all public APIs
- **Documentation**: Google-style docstrings

### Code Formatting

Automated formatting with:

```bash
# Format code
black packrepo/ tests/

# Sort imports  
isort packrepo/ tests/

# Type checking
mypy packrepo/
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add semantic similarity baseline algorithm
fix: resolve token counting accuracy for large files  
docs: update API reference for FastPath v5
test: add integration tests for evaluation harness
refactor: optimize centrality computation performance
```

## 🔍 Review Process

### Pull Request Checklist

Before submitting:

- [ ] **Tests**: All tests pass including new test coverage
- [ ] **Type Checking**: mypy passes without errors
- [ ] **Documentation**: Updated docstrings and user documentation
- [ ] **Performance**: No significant performance regressions
- [ ] **Research Validation**: Algorithm improvements demonstrated empirically

### Review Criteria

Reviewers evaluate:

1. **Technical Quality**: Code correctness, efficiency, and maintainability
2. **Research Rigor**: Theoretical soundness and empirical validation
3. **Documentation**: Clarity and completeness of documentation
4. **Impact**: Significance of contribution to project goals
5. **Compatibility**: Backward compatibility and API stability

### Reviewer Assignment

- **Algorithm Changes**: Require review from research team member
- **Core Infrastructure**: Require review from maintainer
- **Documentation**: Can be reviewed by any contributor
- **Bug Fixes**: Single reviewer approval sufficient

## 🏆 Recognition

### Contributor Acknowledgment

Contributors are recognized through:

- **GitHub Contributions**: Automatic recognition in repository
- **Release Notes**: Significant contributions mentioned in changelog
- **Research Papers**: Co-authorship for substantial research contributions
- **Conference Presentations**: Speaking opportunities for major features

### Research Collaboration

For significant research contributions, we offer:

- **Joint Publications**: Co-authorship opportunities on academic papers
- **Conference Presentations**: Support for presenting work at conferences
- **Research Partnerships**: Collaboration on grant proposals and projects
- **Academic Networking**: Introductions to relevant research communities

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussion
- **Email**: Direct contact for sensitive or complex issues
- **Research Meetings**: Monthly virtual meetings for research discussions

### Maintainer Contact

- **Research Questions**: research@fastpath.ai
- **Technical Issues**: support@fastpath.ai
- **Partnership Inquiries**: partnerships@fastpath.ai

### Response Times

- **Bug Reports**: 2-3 business days
- **Feature Requests**: 1-2 weeks for initial feedback
- **Pull Reviews**: 3-5 business days
- **Research Discussions**: 1 week for detailed technical feedback

## 📄 License

By contributing to FastPath, you agree that your contributions will be licensed under the BSD Zero Clause License.

---

**Thank you for contributing to advancing LLM code analysis research! 🚀**
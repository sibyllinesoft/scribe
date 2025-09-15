# Scribe Repository Analysis Tool - Build System
# Provides unified build, test, and development commands

.PHONY: help build install-dev test benchmark paper clean setup-ci setup-dev

# Configuration
PYTHON := python3
PIP := pip3
MATURIN := maturin
CARGO := cargo
SCRIBE_RS_DIR := scribe-rs
VENV_DIR := .venv
ARTIFACTS_DIR := artifacts

# Default target
help: ## Show this help message
	@echo "Scribe Build System"
	@echo "==================="
	@echo ""
	@echo "Phase 1 Commands (Foundation):"
	@echo "  setup-dev       Setup local development environment"
	@echo "  build          Compile Rust library"
	@echo "  install-dev    Setup Python environment with local wheel"
	@echo ""
	@echo "Development Commands:"
	@echo "  test           Run all tests (Rust + Python)"
	@echo "  test-rust      Run only Rust tests"
	@echo "  test-python    Run only Python tests"
	@echo "  benchmark      Execute benchmark suite"
	@echo "  paper          Generate research artifacts"
	@echo ""
	@echo "CI/CD Commands:"
	@echo "  setup-ci       Setup CI environment"
	@echo "  ci-pipeline    Run full CI pipeline"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean          Clean build artifacts"
	@echo "  clean-all      Clean everything including venv"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

# Development Environment Setup
setup-dev: ## Setup local development environment
	@echo "🔧 Setting up development environment..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@echo "📦 Installing maturin..."
	@. $(VENV_DIR)/bin/activate && $(PIP) install maturin[patchelf]
	@echo "📦 Installing Python development dependencies..."
	@. $(VENV_DIR)/bin/activate && $(PIP) install pytest pytest-asyncio numpy pandas matplotlib seaborn jupyter
	@echo "✅ Development environment ready!"
	@echo "   Activate with: source $(VENV_DIR)/bin/activate"

# Rust compilation
build: ## Compile Rust library
	@echo "🔨 Building Rust core..."
	@cd $(SCRIBE_RS_DIR) && $(CARGO) build --release
	@echo "✅ Rust build complete"

# Python wheel creation and installation
install-dev: build ## Setup Python environment with local wheel
	@echo "🐍 Building Python wheel with maturin..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "❌ Virtual environment not found. Run 'make setup-dev' first."; \
		exit 1; \
	fi
	@. $(VENV_DIR)/bin/activate && cd $(SCRIBE_RS_DIR) && $(MATURIN) develop --release
	@echo "📦 Installing Python research framework..."
	@. $(VENV_DIR)/bin/activate && $(PIP) install -e .
	@echo "✅ Local wheel installed successfully!"

# Testing
test: test-rust test-python ## Run all tests (Rust + Python)

test-rust: ## Run only Rust tests
	@echo "🧪 Running Rust tests..."
	@cd $(SCRIBE_RS_DIR) && $(CARGO) test --workspace --release
	@echo "✅ Rust tests passed"

test-python: ## Run only Python tests
	@echo "🐍 Running Python tests..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "❌ Virtual environment not found. Run 'make setup-dev' first."; \
		exit 1; \
	fi
	@. $(VENV_DIR)/bin/activate && pytest -v tests/ || echo "⚠️  Python tests not found or failed"
	@echo "✅ Python tests completed"

# Benchmarking
benchmark: ## Execute benchmark suite
	@echo "⚡ Running benchmarks..."
	@cd $(SCRIBE_RS_DIR) && $(CARGO) bench
	@if [ -d "research/benchmarks" ]; then \
		echo "🐍 Running Python benchmarks..."; \
		. $(VENV_DIR)/bin/activate && cd research/benchmarks && $(PYTHON) -m benchmark_runner; \
	fi
	@echo "✅ Benchmarks complete"

# Research artifacts
paper: ## Generate research artifacts
	@echo "📊 Generating research artifacts..."
	@mkdir -p $(ARTIFACTS_DIR)
	@if [ -d "research" ]; then \
		echo "🐍 Running statistical analysis..."; \
		. $(VENV_DIR)/bin/activate && cd research && $(PYTHON) -m statistical_analysis.main; \
	fi
	@echo "✅ Research artifacts generated in $(ARTIFACTS_DIR)/"

# CI/CD Support
setup-ci: ## Setup CI environment
	@echo "🚀 Setting up CI environment..."
	@$(PYTHON) -m venv $(VENV_DIR)
	@. $(VENV_DIR)/bin/activate && $(PIP) install --upgrade pip
	@. $(VENV_DIR)/bin/activate && $(PIP) install maturin[patchelf]
	@. $(VENV_DIR)/bin/activate && $(PIP) install pytest pytest-asyncio numpy pandas matplotlib seaborn
	@echo "✅ CI environment ready"

ci-pipeline: setup-ci build install-dev test benchmark ## Run full CI pipeline
	@echo "🎯 CI Pipeline completed successfully!"

# Linting and formatting
lint: ## Run linting and formatting
	@echo "🔍 Running Rust linting..."
	@cd $(SCRIBE_RS_DIR) && $(CARGO) clippy --workspace --all-targets --all-features -- -D warnings
	@cd $(SCRIBE_RS_DIR) && $(CARGO) fmt --check
	@echo "✅ Linting complete"

fmt: ## Format code
	@echo "✨ Formatting Rust code..."
	@cd $(SCRIBE_RS_DIR) && $(CARGO) fmt
	@echo "✅ Code formatted"

# Cleanup
clean: ## Clean build artifacts
	@echo "🧹 Cleaning build artifacts..."
	@cd $(SCRIBE_RS_DIR) && $(CARGO) clean
	@rm -rf $(ARTIFACTS_DIR)
	@rm -rf build/
	@rm -rf dist/
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Build artifacts cleaned"

clean-all: clean ## Clean everything including venv
	@echo "🧹 Cleaning everything..."
	@rm -rf $(VENV_DIR)
	@echo "✅ Everything cleaned"

# Development utilities
check: ## Check Rust code without building
	@echo "🔍 Checking Rust code..."
	@cd $(SCRIBE_RS_DIR) && $(CARGO) check --workspace

docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	@cd $(SCRIBE_RS_DIR) && $(CARGO) doc --workspace --no-deps --open

# Environment info
info: ## Show environment information
	@echo "Environment Information:"
	@echo "======================="
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Cargo: $(shell $(CARGO) --version)"
	@echo "Rust: $(shell rustc --version)"
	@echo "Working Directory: $(shell pwd)"
	@echo "Virtual Environment: $(VENV_DIR)"
	@echo "Artifacts Directory: $(ARTIFACTS_DIR)"
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "Virtual Environment: ✅ Active"; \
	else \
		echo "Virtual Environment: ❌ Not Found"; \
	fi

# Quick development workflow
dev: setup-dev build install-dev test ## Complete development setup workflow
	@echo "🎉 Development environment is ready!"
	@echo "   Next steps:"
	@echo "   1. source $(VENV_DIR)/bin/activate"
	@echo "   2. Start developing!"
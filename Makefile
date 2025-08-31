# PackRepo Makefile - Hermetic Build Operations
# Provides convenient commands for development and CI/CD operations

.PHONY: help build test clean dev ci verify smoke schema lint security deps-check bootstrap

# Default Python and container settings
PYTHON := python3
CONTAINER_TAG := packrepo:dev
ARTIFACTS_DIR := ./artifacts
TEST_REPO := https://github.com/karpathy/nanoGPT

# Help target - default when just running 'make'
help: ## Show this help message
	@echo "PackRepo - Sophisticated Repository Packing System"
	@echo "=================================================="
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Development workflow:"
	@echo "  make bootstrap  # Initial setup"
	@echo "  make dev        # Start development environment" 
	@echo "  make test       # Run tests"
	@echo "  make smoke      # Run smoke tests"
	@echo ""

# Bootstrap - Initial setup
bootstrap: ## Initialize development environment
	@echo "🚀 Bootstrapping PackRepo development environment..."
	@mkdir -p $(ARTIFACTS_DIR) locks spec/schemas tests/{properties,metamorphic,mutation,fuzzing,e2e}
	@if command -v uv >/dev/null 2>&1; then \
		echo "Installing dependencies with uv..."; \
		uv sync; \
	else \
		echo "uv not found, using pip..."; \
		$(PYTHON) -m pip install -r requirements.txt; \
	fi
	@echo "✓ Bootstrap complete"

# Development environment
dev: ## Start development environment with Docker Compose
	@echo "🛠️  Starting PackRepo development environment..."
	docker-compose -f infra/compose.yaml -f infra/compose.override.yaml up --build packrepo-dev

dev-bg: ## Start development environment in background
	docker-compose -f infra/compose.yaml -f infra/compose.override.yaml up -d --build packrepo-dev

dev-stop: ## Stop development environment
	docker-compose -f infra/compose.yaml -f infra/compose.override.yaml down

dev-logs: ## Show development container logs
	docker-compose -f infra/compose.yaml logs -f packrepo-dev

# Build operations
build: ## Build all Docker containers
	@echo "🔨 Building PackRepo containers..."
	docker build -t $(CONTAINER_TAG) -f infra/Dockerfile --target development .
	docker build -t packrepo:ci -f infra/Dockerfile --target ci .
	docker build -t packrepo:prod -f infra/Dockerfile --target production .
	@echo "✓ Build complete"

# Testing operations
test: ## Run full test suite
	@echo "🧪 Running PackRepo test suite..."
	@mkdir -p $(ARTIFACTS_DIR)
	./scripts/ci_full_test.sh

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	$(PYTHON) -m pytest tests/ -v

test-integration: ## Run integration tests
	@echo "Running integration tests..."
	docker-compose -f infra/compose.yaml --profile integration up --abort-on-container-exit

test-watch: ## Run tests in watch mode
	docker-compose -f infra/compose.yaml -f infra/compose.override.yaml --profile watch up

# Verification operations
verify: ## Verify pack format and constraints
	@echo "✅ Running pack verification..."
	$(PYTHON) scripts/pack_verify.py --write-schema spec/index.schema.json
	@if [ -d "logs" ]; then \
		$(PYTHON) scripts/pack_verify.py --packs logs/ --schema spec/index.schema.json; \
	else \
		echo "No logs directory found - run smoke tests first"; \
	fi

smoke: ## Run hermetic smoke tests
	@echo "💨 Running hermetic smoke tests..."
	./scripts/spinup_smoke.sh --repo $(TEST_REPO) --budget 50000 --tokenizer cl100k --no-llm

smoke-self: ## Run smoke tests on current repository
	@echo "💨 Running self smoke tests..."
	./scripts/spinup_smoke.sh --budget 50000 --tokenizer cl100k --no-llm

# Schema operations
schema: ## Generate and validate JSON schemas
	@echo "📋 Generating schemas..."
	@mkdir -p spec
	$(PYTHON) scripts/pack_verify.py --write-schema spec/index.schema.json
	@echo "✓ Schema generation complete"

# Code quality operations
lint: ## Run code linting and formatting
	@echo "🔍 Running linting..."
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check . --fix; \
		ruff format .; \
	else \
		echo "ruff not found, installing..."; \
		$(PYTHON) -m pip install ruff; \
		ruff check . --fix; \
		ruff format .; \
	fi
	@echo "✓ Linting complete"

lint-check: ## Check code formatting without fixing
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check .; \
		ruff format --check .; \
	else \
		echo "ruff not found - run 'make lint' first"; \
		exit 1; \
	fi

typecheck: ## Run type checking
	@echo "🏷️  Running type checks..."
	@if command -v mypy >/dev/null 2>&1; then \
		mypy packrepo/; \
	else \
		echo "mypy not found, installing..."; \
		$(PYTHON) -m pip install mypy; \
		mypy packrepo/; \
	fi

# Security operations
security: ## Run security scans
	@echo "🔒 Running security scans..."
	@mkdir -p $(ARTIFACTS_DIR)
	./scripts/scan_secrets.py --out $(ARTIFACTS_DIR)/secret_scan.json
	@if command -v bandit >/dev/null 2>&1; then \
		bandit -r packrepo/ -f json -o $(ARTIFACTS_DIR)/bandit_results.json; \
	else \
		echo "bandit not found, installing..."; \
		$(PYTHON) -m pip install bandit; \
		bandit -r packrepo/ -f json -o $(ARTIFACTS_DIR)/bandit_results.json; \
	fi
	@echo "✓ Security scan complete"

# Dependency operations
deps-check: ## Check for dependency vulnerabilities
	@echo "📦 Checking dependencies..."
	@if command -v safety >/dev/null 2>&1; then \
		safety check --json --output $(ARTIFACTS_DIR)/safety_results.json; \
	else \
		echo "safety not found, installing..."; \
		$(PYTHON) -m pip install safety; \
		safety check --json --output $(ARTIFACTS_DIR)/safety_results.json; \
	fi

deps-update: ## Update dependencies
	@echo "📦 Updating dependencies..."
	@if command -v uv >/dev/null 2>&1; then \
		uv lock --upgrade; \
	else \
		$(PYTHON) -m pip list --outdated; \
		echo "Consider using uv for better dependency management"; \
	fi

# CI operations
ci: ## Run complete CI pipeline
	@echo "🚀 Running complete CI pipeline..."
	@$(MAKE) clean
	@$(MAKE) build
	@$(MAKE) test
	@$(MAKE) verify
	@$(MAKE) security
	@$(MAKE) smoke-self
	@echo "✅ CI pipeline complete"

ci-docker: ## Run CI in Docker containers
	@echo "🐳 Running CI in Docker..."
	docker-compose -f infra/compose.yaml --profile ci up --abort-on-container-exit

# Deployment operations
deploy-staging: ## Deploy to staging environment
	@echo "🚀 Deploying to staging..."
	docker-compose -f infra/compose.yaml --profile prod up -d
	@echo "✓ Staging deployment complete"

# Cleanup operations
clean: ## Clean up generated files and containers
	@echo "🧹 Cleaning up..."
	rm -rf $(ARTIFACTS_DIR)/* logs/* locks/*.bak
	docker-compose -f infra/compose.yaml down --remove-orphans --volumes
	docker system prune -f
	@echo "✓ Cleanup complete"

clean-all: ## Clean everything including Docker images
	@$(MAKE) clean
	docker rmi -f $(CONTAINER_TAG) packrepo:ci packrepo:prod || true
	docker system prune -a -f

# Documentation operations
docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	@mkdir -p docs
	@echo "# PackRepo Documentation" > docs/README.md
	@echo "" >> docs/README.md
	@echo "Generated on: $$(date)" >> docs/README.md
	@echo "" >> docs/README.md
	@echo "## Schema Documentation" >> docs/README.md
	@if [ -f "spec/index.schema.json" ]; then \
		echo "See [index.schema.json](../spec/index.schema.json)" >> docs/README.md; \
	fi
	@echo "✓ Documentation generated"

# Status and information
status: ## Show project status
	@echo "📊 PackRepo Status"
	@echo "=================="
	@echo "Project root: $$(pwd)"
	@echo "Python version: $$($(PYTHON) --version 2>&1)"
	@echo "Docker version: $$(docker --version 2>&1 || echo 'Docker not available')"
	@echo ""
	@echo "Directory structure:"
	@find . -name "*.py" -type f | head -10
	@echo ""
	@echo "Containers:"
	@docker ps --filter "name=packrepo" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || echo "No containers running"
	@echo ""
	@if [ -d "$(ARTIFACTS_DIR)" ]; then \
		echo "Recent artifacts:"; \
		ls -la $(ARTIFACTS_DIR) | head -5; \
	fi

# Quick operations
quick-test: bootstrap lint typecheck test-unit ## Run quick development tests
	@echo "✅ Quick tests complete"

full-test: bootstrap build test verify security smoke-self ## Run comprehensive tests
	@echo "✅ Full test suite complete"

# Default target
.DEFAULT_GOAL := help
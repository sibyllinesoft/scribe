# Makefile for Scribe E2E Tests
# Handles compilation dependencies and multiple test targets

.PHONY: all help clean install build test test-lib test-webui test-cli test-watch test-headed test-debug deps check-deps check-rust check-bun

# Default target
all: deps build test

# Help target
help:
	@echo "Scribe E2E Test Suite"
	@echo ""
	@echo "Available targets:"
	@echo "  help          Show this help message"
	@echo "  deps          Install all dependencies"
	@echo "  build         Build Scribe binary"
	@echo "  test          Run all tests"
	@echo "  test-lib      Run library unit tests"
	@echo "  test-webui    Run web UI e2e tests"
	@echo "  test-cli      Run CLI parameter tests"
	@echo "  test-watch    Run tests in watch mode"
	@echo "  test-headed   Run Playwright tests with visible browser"
	@echo "  test-debug    Run Playwright tests in debug mode"
	@echo "  install       Install dependencies only"
	@echo "  clean         Clean build artifacts and node_modules"
	@echo "  check-deps    Check if required tools are installed"
	@echo ""
	@echo "Environment variables:"
	@echo "  VERBOSE=1     Enable verbose output"
	@echo "  PORT=8080     Override default test port"
	@echo "  HEADLESS=0    Run browsers in headed mode"

# Variables
SCRIBE_DIR := ../scribe-rs
SCRIBE_BIN := $(SCRIBE_DIR)/target/debug/scribe-web
NODE_MODULES := node_modules
PACKAGE_JSON := package.json
PLAYWRIGHT_CONFIG := playwright.config.js

# Check if verbose mode is enabled
ifdef VERBOSE
	VERBOSE_FLAG := --verbose
	CARGO_VERBOSE := --verbose
else
	VERBOSE_FLAG := 
	CARGO_VERBOSE := 
endif

# Check if headed mode is requested
ifdef HEADLESS
ifeq ($(HEADLESS),0)
	HEADED_FLAG := --headed
endif
endif

# Default port - use 9876 for CI, 8080 for local
PORT ?= $(if $(CI),9876,8080)

# Dependency checks
check-deps: check-rust check-bun
	@echo "✓ All dependencies are available"

check-rust:
	@command -v cargo >/dev/null 2>&1 || { \
		echo "Error: Rust/Cargo not found. Please install Rust first."; \
		echo "Visit: https://rustup.rs/"; \
		exit 1; \
	}
	@command -v rustc >/dev/null 2>&1 || { \
		echo "Error: Rust compiler not found."; \
		exit 1; \
	}
	@echo "✓ Rust toolchain found"

check-bun:
	@command -v bun >/dev/null 2>&1 || { \
		echo "Error: Bun not found. Please install Bun first."; \
		echo "Visit: https://bun.sh/"; \
		exit 1; \
	}
	@echo "✓ Bun runtime found"

# Install dependencies
install: check-deps $(NODE_MODULES)

$(NODE_MODULES): $(PACKAGE_JSON)
	@echo "Installing Node.js dependencies..."
	bun install
	@echo "Installing Playwright browsers..."
	bun run playwright:install
	@touch $(NODE_MODULES)

deps: install
	@echo "✓ All dependencies installed"

# Build Scribe binary
build: check-rust $(SCRIBE_BIN)

$(SCRIBE_BIN): $(SCRIBE_DIR)/Cargo.toml $(shell find $(SCRIBE_DIR) -name "*.rs" 2>/dev/null)
	@echo "Building Scribe binary..."
	cd $(SCRIBE_DIR) && cargo build -p scribe-webservice --bin scribe-web $(CARGO_VERBOSE)
	@echo "✓ Scribe binary built successfully"

# Test targets
test: deps build test-lib test-cli test-webui
	@echo "✓ All tests completed successfully"

test-lib: deps
	@echo "Running library unit tests..."
	bun test lib/ $(VERBOSE_FLAG)
	@echo "✓ Library tests passed"

test-webui: deps build $(PLAYWRIGHT_CONFIG)
	@echo "Running web UI e2e tests..."
	bun run playwright test $(HEADED_FLAG) $(VERBOSE_FLAG)
	@echo "✓ Web UI tests passed"

test-cli: deps build
	@echo "Running CLI parameter tests..."
	bun test cli/ $(VERBOSE_FLAG)
	@echo "✓ CLI tests passed"

# Development targets
test-watch: deps
	@echo "Running tests in watch mode..."
	bun test --watch $(VERBOSE_FLAG)

test-headed: deps build
	@echo "Running Playwright tests in headed mode..."
	bun run playwright test --headed $(VERBOSE_FLAG)

test-debug: deps build
	@echo "Running Playwright tests in debug mode..."
	bun run playwright test --debug

test-ui: deps build
	@echo "Running Playwright tests with UI..."
	bun run playwright test --ui

# Performance and stress tests
test-perf: deps build
	@echo "Running performance tests..."
	bun test --grep "Performance" $(VERBOSE_FLAG)

test-stress: deps build
	@echo "Running stress tests..."
	bun test --grep "stress|large" $(VERBOSE_FLAG)

# Specific test files
test-arborist: deps build
	@echo "Running React Arborist tests..."
	bun run playwright test webui/react-arborist.spec.js $(HEADED_FLAG) $(VERBOSE_FLAG)

test-cli-params: deps build  
	@echo "Running CLI parameter tests..."
	bun test cli/cli-parameters.spec.js $(VERBOSE_FLAG)

test-tree-building: deps
	@echo "Running tree building tests..."
	bun test lib/tree-building.test.js $(VERBOSE_FLAG)

# Lint and format
lint: deps
	@echo "Running ESLint..."
	@if [ -f .eslintrc.js ] || [ -f .eslintrc.json ]; then \
		bunx eslint . --ext .js,.mjs; \
	else \
		echo "No ESLint config found, skipping..."; \
	fi

format: deps
	@echo "Running Prettier..."
	@if [ -f .prettierrc ] || [ -f prettier.config.js ]; then \
		bunx prettier --write . --ignore-path .gitignore; \
	else \
		echo "No Prettier config found, skipping..."; \
	fi

# Quality checks
typecheck: deps
	@echo "Running TypeScript type checking..."
	@if [ -f tsconfig.json ]; then \
		bunx tsc --noEmit; \
	else \
		echo "No TypeScript config found, skipping..."; \
	fi

# Utility targets
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(NODE_MODULES)
	rm -rf playwright-report
	rm -rf test-results
	cd $(SCRIBE_DIR) && cargo clean
	@echo "✓ Clean completed"

clean-cache:
	@echo "Cleaning test caches..."
	rm -rf .bun-cache
	rm -rf playwright-report
	rm -rf test-results
	@echo "✓ Cache cleaned"

# Development server
dev-server: build
	@echo "Starting Scribe development server on port $(PORT)..."
	cd $(SCRIBE_DIR) && cargo run -p scribe-webservice --bin scribe-web -- . --port $(PORT) --no-browser --no-auto-shutdown

# CI/CD targets
ci: check-deps install build test

ci-headless: HEADLESS := 1
ci-headless: ci

# Docker targets (if Docker is available)
docker-test:
	@if command -v docker >/dev/null 2>&1; then \
		echo "Running tests in Docker container..."; \
		docker run --rm -v $(PWD):/workspace -w /workspace \
			-e CI=true \
			oven/bun:1 \
			make ci-headless; \
	else \
		echo "Docker not available, running tests locally..."; \
		make ci-headless; \
	fi

# Report generation
report: test-webui
	@echo "Generating test reports..."
	@if [ -d playwright-report ]; then \
		echo "✓ Playwright HTML report available at: playwright-report/index.html"; \
		echo "✓ Run 'bun run playwright show-report' to view"; \
	fi

# Debug targets
debug-deps:
	@echo "=== Dependency Information ==="
	@echo "Node version: $$(node --version 2>/dev/null || echo 'Not found')"
	@echo "Bun version: $$(bun --version 2>/dev/null || echo 'Not found')"
	@echo "Cargo version: $$(cargo --version 2>/dev/null || echo 'Not found')"
	@echo "Rust version: $$(rustc --version 2>/dev/null || echo 'Not found')"
	@echo "=== File Status ==="
	@echo "Package.json exists: $$([ -f $(PACKAGE_JSON) ] && echo 'Yes' || echo 'No')"
	@echo "Node modules exists: $$([ -d $(NODE_MODULES) ] && echo 'Yes' || echo 'No')"
	@echo "Scribe binary exists: $$([ -f $(SCRIBE_BIN) ] && echo 'Yes' || echo 'No')"
	@echo "Playwright config exists: $$([ -f $(PLAYWRIGHT_CONFIG) ] && echo 'Yes' || echo 'No')"

debug-scribe:
	@echo "=== Scribe Build Information ==="
	cd $(SCRIBE_DIR) && cargo check -p scribe-webservice --bin scribe-web
	@echo "✓ Scribe binary compiles successfully"
	cd $(SCRIBE_DIR) && cargo run -p scribe-webservice --bin scribe-web -- --version
	@echo "✓ Scribe binary runs successfully"

# Help for specific test categories
help-lib:
	@echo "Library Test Targets:"
	@echo "  test-lib           Run all library unit tests"
	@echo "  test-tree-building Run tree building tests specifically"
	@echo "  test-perf          Run performance tests"

help-webui:
	@echo "Web UI Test Targets:"
	@echo "  test-webui         Run all web UI e2e tests"
	@echo "  test-arborist      Run React Arborist tests specifically"
	@echo "  test-headed        Run tests with visible browser"
	@echo "  test-debug         Run tests in debug mode"
	@echo "  test-ui            Run tests with Playwright UI"

help-cli:
	@echo "CLI Test Targets:"
	@echo "  test-cli           Run all CLI parameter tests"
	@echo "  test-cli-params    Run CLI parameter tests specifically"
	@echo "  dev-server         Start development server for manual testing"

# Validate test setup
validate: check-deps
	@echo "Validating test setup..."
	@echo "✓ Checking if package.json is valid..."
	@bun run --silent --eval "JSON.parse(require('fs').readFileSync('package.json', 'utf8'))"
	@echo "✓ Checking if Playwright config is valid..."
	@node -e "require('./playwright.config.js')" 2>/dev/null || echo "⚠ Playwright config may have issues"
	@echo "✓ Checking test file syntax..."
	@find lib webui cli -name "*.js" -exec node -c {} \; 2>/dev/null || echo "⚠ Some test files may have syntax issues"
	@echo "✓ Test setup validation completed"

# Install only Playwright dependencies
install-playwright: check-bun
	@echo "Installing Playwright dependencies..."
	bun add -D @playwright/test playwright
	bun run playwright install

# Install only Bun test dependencies  
install-bun-deps: check-bun
	@echo "Installing Bun test dependencies..."
	bun add -D @types/node

# Quick smoke test
smoke: deps build
	@echo "Running smoke tests..."
	timeout 30 bun test lib/tree-building.test.js --grep "should create tree from flat file list"
	@echo "✓ Smoke test passed"

# Full test suite with timing
test-timed: deps build
	@echo "Running timed test suite..."
	@start_time=$$(date +%s); \
	make test; \
	end_time=$$(date +%s); \
	duration=$$((end_time - start_time)); \
	echo "✓ Full test suite completed in $${duration} seconds"
# Scribe E2E Tests

This directory contains consolidated end-to-end tests for the Scribe project using Bun and Playwright.

## Directory Structure

```
tests/
├── lib/               # Library unit tests (Bun)
├── webui/             # Web UI e2e tests (Playwright)
├── cli/               # CLI e2e tests (Playwright + process spawning)
├── package.json       # Test dependencies
├── playwright.config.js  # Playwright configuration
└── README.md         # This file
```

## Test Categories

### Library Tests (`lib/`)
- **Framework**: Bun Test
- **Purpose**: Unit tests for core library functionality
- **Files**: Tree building logic, checkbox state management, performance tests
- **Run with**: `bun test lib/`

### Web UI Tests (`webui/`)
- **Framework**: Playwright
- **Purpose**: End-to-end tests for React Arborist integration and web interface
- **Files**: React component behavior, user interactions, tree operations
- **Run with**: `bun test:e2e` or `playwright test`

### CLI Tests (`cli/`)
- **Framework**: Playwright + Node.js process spawning
- **Purpose**: End-to-end tests for CLI parameters and web interface integration
- **Files**: All CLI parameter combinations, error handling, help/version
- **Run with**: `bun test cli/`

## Setup

### Prerequisites
- **Rust toolchain** (cargo, rustc)
- **Bun runtime** (>= 1.0.0)

### Quick Start
```bash
cd tests
make help           # Show all available targets
make deps           # Install all dependencies
make check-deps     # Verify prerequisites
```

## Running Tests

### Using Makefile (Recommended)
```bash
# All tests (with automatic dependency management)
make test

# Individual test categories
make test-lib       # Library unit tests
make test-webui     # Web UI e2e tests  
make test-cli       # CLI parameter tests

# Development targets
make test-watch     # Watch mode
make test-headed    # Visual browser mode
make test-debug     # Debug mode
make test-ui        # Playwright UI mode
```

### Direct Commands (Alternative)
```bash
# Library unit tests only (no compilation required)
bun test lib/

# All Bun tests
bun test

# Playwright tests (requires Scribe binary)
bun run playwright test
```

### Environment Variables
```bash
# Enable verbose output
VERBOSE=1 make test

# Run with headed browsers
HEADLESS=0 make test-webui

# Use custom port
PORT=9000 make dev-server
```

## Test Features

### React Arborist Testing
- **Tree rendering and structure**
- **Folder expansion/collapse**
- **File selection with checkboxes**
- **Keyboard navigation**
- **Large tree performance**
- **Error state handling**
- **Accessibility features**

### CLI Parameter Testing
- **Port and host configuration**
- **Token budget limits**
- **Max file size constraints**
- **Test file inclusion/exclusion**
- **Parameter combinations**
- **Error handling and validation**
- **Help and version information**

### Library Testing
- **Tree building from file lists**
- **Checkbox state management**
- **File icon handling**
- **Performance with large datasets**
- **Edge case handling**

## Configuration

### Playwright Config
- **Base URL**: `http://localhost:8080`
- **Browsers**: Chrome, Firefox, Safari, Mobile browsers
- **Auto-start**: Scribe web server before tests
- **Retry**: Enabled on CI
- **Reporters**: HTML reports

### Test Data
- Uses real Scribe project structure for realistic testing
- Generates synthetic large datasets for performance testing
- Handles edge cases like special characters and malformed data

## Debugging React Arborist Issues

The React Arborist tests specifically target common issues:

1. **Tree State Management**: Tests checkbox synchronization and selection state
2. **Performance**: Large tree handling and rapid state changes
3. **User Interactions**: Keyboard navigation, drag & drop, search/filtering
4. **Error Recovery**: Network failures, malformed data, component errors
5. **Accessibility**: ARIA attributes, focus management, screen reader support

## CI/CD Integration

The test suite is designed for CI/CD environments:
- Headless browser execution
- Timeout configurations
- Retry logic for flaky tests
- Proper cleanup of spawned processes
- Detailed error reporting

## Troubleshooting

### Compilation Issues

If you encounter Rust compilation errors when running web UI or CLI tests:

1. **Library tests work independently**: `make test-lib` doesn't require Scribe binary
2. **Check Scribe codebase**: Ensure the main Scribe project compiles
3. **Build manually**: `cd ../scribe-rs && cargo build -p scribe-webservice --bin scribe-web`
4. **Debug build issues**: `make debug-scribe` for detailed build information

### Common Issues

1. **Server startup timeout**: Increase timeout in playwright config
2. **Port conflicts**: Tests use different ports (8081-8093)  
3. **Process cleanup**: Tests properly kill spawned Scribe processes
4. **Browser installation**: Run `make install` or `bun run playwright:install`
5. **Missing dependencies**: Run `make check-deps` to verify prerequisites

### Debug Commands
```bash
# Check prerequisites
make check-deps

# Debug dependency information  
make debug-deps

# Build Scribe manually
cd ../scribe-rs && cargo build -p scribe-webservice --bin scribe-web

# Test server startup manually
cd ../scribe-rs && cargo run -p scribe-webservice --bin scribe-web -- . --port 8080 --no-browser

# Run single test file
bun test webui/react-arborist.spec.js

# Run with verbose output
VERBOSE=1 make test-lib
```

### Test Isolation

The test suite is designed with multiple levels of isolation:

- **Library tests**: No external dependencies, test core logic
- **CLI tests**: Require Scribe binary, test command-line interface
- **Web UI tests**: Require both Scribe binary and browser automation

You can run library tests even if the Scribe binary doesn't compile.
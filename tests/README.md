# Scribe Test Suite

This directory contains the comprehensive, organized test suite for Scribe - an intelligent repository analysis tool for LLM code consumption.

## Overview

The test suite has been completely reorganized from 38+ chaotic test files into a clean, modular structure that follows Python testing best practices. All important test cases have been consolidated and enhanced with better coverage, error handling, and edge case testing.

## Test Structure

### Core Test Modules

- **`test_glob_patterns.py`** - Comprehensive tests for glob pattern matching functionality
  - Comma-separated pattern parsing
  - Wildcard and recursive glob matching (`**`)
  - Include/exclude pattern logic
  - Unicode and edge case handling

- **`test_file_analysis.py`** - File analysis and processing functionality
  - Binary file detection
  - GitIgnore pattern handling  
  - File decision logic (size, binary, ignored files)
  - File collection and filtering
  - Content loading and token estimation

- **`test_cli.py`** - Command-line interface testing
  - Argument parsing and validation
  - Main function integration testing
  - Remote repository handling
  - Editor mode functionality
  - Error handling and edge cases

- **`test_output_formats.py`** - Output format generation
  - CXML format for LLM consumption
  - Repomix format compatibility
  - HTML format with modern styling
  - File icon detection
  - Content escaping and safety

- **`test_edge_cases.py`** - Edge cases, error conditions, and boundary testing
  - File system edge cases (permissions, symlinks, deep paths)
  - Network and repository access issues
  - Memory and performance boundaries
  - Input validation and sanitization
  - Platform-specific compatibility

### Supporting Files

- **`conftest.py`** - Shared pytest fixtures and configuration
- **`__init__.py`** - Test package initialization

### Configuration

- **`pytest.ini`** - Pytest configuration with markers and options
- **`run_tests.py`** - Comprehensive test runner script

## Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py

# Run quick tests (excluding slow/network tests)
python run_tests.py --quick

# Run with coverage report
python run_tests.py --coverage-html

# Run specific module
python run_tests.py --module glob_patterns
```

### Using pytest directly

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_glob_patterns.py

# Run with coverage
pytest --cov=scribe --cov-report=html

# Run excluding slow tests
pytest -m "not slow"
```

### Test Categories

Tests are organized with markers for different categories:

- **`slow`** - Tests that take significant time to run
- **`integration`** - Integration tests that test multiple components
- **`network`** - Tests that require network access
- **`platform_specific`** - Platform-dependent tests

## Test Coverage

The test suite achieves comprehensive coverage of Scribe's functionality:

- **Glob Pattern Matching** - All glob patterns, edge cases, and recursive patterns
- **File Analysis** - Binary detection, size limits, gitignore patterns
- **Repository Operations** - Git integration, remote repositories, file collection
- **Output Formats** - All three output formats with proper escaping and formatting
- **Command Line Interface** - All CLI options, error handling, edge cases
- **Edge Cases** - File system issues, network problems, input validation

### Coverage Goals

- **Line Coverage**: Target 85%+ overall coverage
- **Branch Coverage**: Comprehensive testing of all code paths
- **Edge Case Coverage**: Extensive testing of error conditions and boundary cases

## Test Organization Principles

### Clean Architecture

1. **Modular Organization** - Each test module focuses on a specific area of functionality
2. **Clear Naming** - Test classes and methods have descriptive names
3. **Comprehensive Fixtures** - Reusable test data and mock objects
4. **Proper Isolation** - Tests don't depend on each other or external state

### Quality Standards

1. **Descriptive Test Names** - Each test clearly describes what it validates
2. **Comprehensive Assertions** - Tests validate both positive and negative cases
3. **Mock Usage** - External dependencies are properly mocked
4. **Error Testing** - Error conditions and edge cases are thoroughly tested

### Best Practices

1. **DRY Principle** - Common test code is factored into fixtures and utilities
2. **Fast Tests** - Most tests run quickly; slow tests are marked appropriately
3. **Deterministic** - Tests produce consistent results across runs
4. **Platform Agnostic** - Tests work across different operating systems

## Migration from Chaos

This organized test suite replaces the previous chaotic collection of 38+ test files with names like:
- `test_victory_85_percent.py`
- `test_absolute_final_85.py`
- `test_final_mega.py`
- `test_grind_to_85.py`

All important test cases from these files have been:
1. **Analyzed** for unique functionality
2. **Consolidated** into appropriate modules
3. **Enhanced** with better error handling
4. **Organized** following Python testing conventions
5. **Documented** with clear descriptions

## Contributing to Tests

When adding new tests:

1. **Choose the Right Module** - Add tests to the most appropriate existing module
2. **Follow Naming Conventions** - Use descriptive test names that explain the scenario
3. **Use Fixtures** - Leverage existing fixtures from `conftest.py`
4. **Add Markers** - Mark slow or platform-specific tests appropriately
5. **Test Edge Cases** - Include both success and failure scenarios
6. **Document Complex Tests** - Add docstrings for non-obvious test logic

## Example Test Pattern

```python
class TestFeatureGroup:
    """Test a specific group of related functionality."""
    
    def test_basic_functionality(self):
        """Test the happy path scenario."""
        # Arrange
        # Act  
        # Assert
        
    def test_edge_case_handling(self):
        """Test handling of edge cases."""
        # Test boundary conditions
        
    def test_error_conditions(self):
        """Test error handling and recovery."""
        # Test various error scenarios
```

## Performance Testing

The test suite includes performance-conscious tests:

- **Memory Usage** - Tests don't consume excessive memory
- **Execution Time** - Fast tests run in milliseconds
- **Scalability** - Tests validate handling of large inputs
- **Resource Cleanup** - Temporary files and resources are properly cleaned up

## Continuous Integration

The test suite is designed for CI/CD environments:

- **Parallel Execution** - Tests can run in parallel safely  
- **Environment Independence** - No dependencies on specific system configuration
- **Clear Exit Codes** - Proper success/failure reporting
- **Comprehensive Coverage** - High confidence in code quality

This organized test suite provides a solid foundation for maintaining and enhancing Scribe's functionality while ensuring high code quality and reliability.
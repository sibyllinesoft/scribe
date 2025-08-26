# FastPath Utilities

This directory contains utility modules that centralize common functionality used across the FastPath system, reducing code duplication and providing consistent behavior.

## Modules

### `entry_points.py` - Entry Point Processing

**Purpose**: Centralized processing and normalization of entry point specifications.

**Key Features**:
- `EntryPointConverter.normalize_entry_points()` - Convert mixed string/EntryPointSpec inputs to standardized format
- `EntryPointConverter.validate_entry_points()` - Validate entry point specifications  
- `EntryPointConverter.get_total_weight()` - Calculate total weight of entry points
- `EntryPointConverter.filter_by_file_patterns()` - Filter entry points by file patterns

**Replaces duplicate code from**:
- `types.py` - ScribeConfig.with_entry_points() and with_diffs()
- `personalized_centrality.py` - create_from_entry_points()
- `diff_packer.py` - Entry point processing logic

### `file_patterns.py` - File Pattern Matching

**Purpose**: Standardized file pattern matching with multiple strategies.

**Key Features**:
- `FilePatternMatcher.matches()` - Match files using exact, suffix, glob, or regex patterns
- `FilePatternMatcher.filter_files()` - Filter file lists with include/exclude patterns
- `FilePatternMatcher.find_symbol_in_file()` - Find functions/classes in source files
- `FilePatternMatcher.is_code_file()` - Detect if file is source code
- Built-in pattern collections for ignore patterns, documentation files, etc.

**Replaces duplicate code from**:
- `personalized_centrality.py` - _file_matches() method
- `fast_scan.py` - File type detection logic
- Various file filtering patterns across the system

### `error_handling.py` - Standardized Error Handling

**Purpose**: Consistent error handling patterns with fallbacks and logging.

**Key Features**:
- `ErrorHandler.safe_execute()` - Execute operations returning Result<T, E> 
- `ErrorHandler.handle_with_fallback()` - Execute with fallback values on error
- `ErrorHandler.with_retry()` - Retry logic with backoff
- `ErrorHandler.safe_file_operation()` - Safe file operations with standard error handling
- `Result<T, E>` type for explicit error handling without exceptions

**Replaces duplicate code from**:
- `fast_scan.py` - File operation error handling
- `personalized_centrality.py` - File reading error handling  
- `diff_packer.py` - Subprocess error handling
- Various try/catch patterns across the system

## Usage Example

```python
from packrepo.fastpath.utils import EntryPointConverter, FilePatternMatcher, ErrorHandler

# Entry point processing
converter = EntryPointConverter()
entry_points = converter.normalize_entry_points([
    "main.py",
    EntryPointSpec(file_path="src/app.py", function_name="main")
])

# File pattern matching  
matcher = FilePatternMatcher()
python_files = matcher.filter_files(all_files, include_patterns=["*.py"])
has_main_func = matcher.find_symbol_in_file("app.py", function_name="main")

# Error handling
handler = ErrorHandler("MyComponent") 
result = handler.safe_execute(lambda: risky_operation())
if result.is_success:
    print(f"Success: {result.value}")
else:
    print(f"Failed: {result.error}")
```

## Benefits

1. **Consistency**: Standardized behavior across the FastPath system
2. **Maintainability**: Single source of truth for common patterns
3. **Reliability**: Well-tested utilities with proper error handling
4. **Extensibility**: Easy to add new functionality in centralized location
5. **Documentation**: Clear interfaces with comprehensive documentation

## Migration Path

These utilities are designed to be drop-in replacements for existing duplicate code:

1. Import the appropriate utility class
2. Replace the duplicate code with utility method calls  
3. Remove the old duplicate implementation
4. Gain improved error handling and validation as a bonus

The utilities maintain backward compatibility with existing interfaces while providing enhanced functionality.
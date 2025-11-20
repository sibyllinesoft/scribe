# scribe-patterns

Advanced pattern matching and filtering for Scribe repository analysis.

## Overview

`scribe-patterns` provides sophisticated pattern matching capabilities for file selection, filtering, and search operations. It handles glob patterns, regex matching, `.gitignore` semantics, and custom ignore rules with high performance and correct edge case handling.

## Key Features

### Glob Pattern Matching
- **Standard glob syntax**: `*`, `**`, `?`, `[abc]`, `{a,b,c}`
- **Directory-aware matching**: Handles `**/` for recursive directory traversal
- **Negative patterns**: `!pattern` to exclude specific files
- **Case sensitivity control**: Case-insensitive matching on Windows by default

### Gitignore Semantics
- **`.gitignore` parsing**: Full compatibility with Git's ignore rules
- **Directory negation**: Properly handles `!` negation patterns
- **Relative vs absolute paths**: Distinguishes `/pattern` from `pattern`
- **Trailing slashes**: Directory-only patterns with `/`
- **Comment support**: Lines starting with `#` are ignored

### Custom Ignore Files
- **`.scribeignore`**: Scribe-specific ignore patterns
- **Multiple ignore files**: Hierarchical ignore file processing
- **Override precedence**: Later patterns override earlier ones
- **Inheritance**: Child directories inherit parent ignore rules

### Performance Optimizations
- **Compiled pattern sets**: Pre-compile globs into efficient matchers
- **Aho-Corasick for literals**: Fast multi-pattern matching for literal strings
- **Regex caching**: Compiled regex patterns are cached
- **Early returns**: Short-circuit evaluation for common cases

## Architecture

```
Pattern Input → Parser → Compiled Matcher → Match Engine
     ↓            ↓            ↓                ↓
 Glob/Regex   Validate   globset/regex    Apply to Paths
  Strings     Syntax     Compilation       Fast Matching
```

### Core Components

#### `PatternSet`
Collection of patterns with unified matching interface:
- **Globs**: File name patterns like `*.rs`, `**/*.py`
- **Regex**: Complex patterns using regular expressions
- **Literals**: Exact string matches (optimized with Aho-Corasick)
- **Negations**: Exclude patterns that override includes

#### `IgnoreBuilder`
Constructs ignore rule sets from multiple sources:
- **`.gitignore` files**: Standard Git ignore semantics
- **`.scribeignore` files**: Scribe-specific patterns
- **Custom patterns**: Programmatically added rules
- **Precedence handling**: Correct override behavior

#### `PathMatcher`
Efficient path matching against pattern sets:
- **Compiled matchers**: Pre-compiled globset for performance
- **Path normalization**: Handles Windows vs Unix path separators
- **Absolute vs relative**: Correct matching for both path types
- **Directory detection**: Special handling for directory patterns

#### `PatternParser`
Parses and validates pattern syntax:
- **Glob expansion**: Converts globs to regex when needed
- **Escape sequence handling**: Properly handles `\*`, `\?`, etc.
- **Error reporting**: Clear error messages for invalid patterns
- **Syntax validation**: Detects malformed patterns early

## Usage

### Basic Glob Matching

```rust
use scribe_patterns::{PatternSet, PathMatcher};

let patterns = PatternSet::from_globs(vec![
    "**/*.rs",           // All Rust files
    "**/*.py",           // All Python files
    "!**/*_test.py",     // Except test files
])?;

let matcher = PathMatcher::new(patterns);

assert!(matcher.is_match("src/main.rs"));
assert!(matcher.is_match("lib/utils.py"));
assert!(!matcher.is_match("lib/utils_test.py")); // Negated
```

### Gitignore-Style Filtering

```rust
use scribe_patterns::IgnoreBuilder;

let mut builder = IgnoreBuilder::new("/path/to/repo");
builder.add_gitignore(".gitignore")?;
builder.add_custom("target/**")?;    // Exclude Rust build directory
builder.add_custom("!target/debug/important.txt")?; // But include this file

let ignore = builder.build()?;

for entry in walkdir::WalkDir::new("/path/to/repo") {
    let entry = entry?;
    if ignore.matched(entry.path(), entry.file_type().is_dir()).is_ignore() {
        continue; // Skip ignored files
    }
    // Process file
}
```

### Multiple Pattern Sets

```rust
use scribe_patterns::{PatternSet, Matcher};

// Include patterns
let include = PatternSet::from_globs(vec![
    "src/**/*.rs",
    "lib/**/*.rs",
])?;

// Exclude patterns
let exclude = PatternSet::from_globs(vec![
    "**/target/**",
    "**/*.bak",
])?;

let matcher = Matcher::new()
    .include(include)
    .exclude(exclude);

// File must match include AND not match exclude
if matcher.should_include("src/utils.rs") {
    // Process file
}
```

### Regex Patterns

```rust
use scribe_patterns::PatternSet;

let patterns = PatternSet::from_regex(vec![
    r".*_test\.(rs|py)$",     // Test files in Rust or Python
    r"^src/.*/mod\.rs$",      // All mod.rs files in src
])?;

assert!(patterns.is_match("src/utils/mod.rs"));
assert!(patterns.is_match("lib/parser_test.py"));
```

### Case-Insensitive Matching

```rust
use scribe_patterns::{PatternSet, MatchOptions};

let patterns = PatternSet::from_globs(vec!["*.TXT", "*.Md"])?;

let options = MatchOptions {
    case_sensitive: false,
    ..Default::default()
};

let matcher = PathMatcher::new(patterns).with_options(options);

assert!(matcher.is_match("readme.md"));  // Matches *.Md
assert!(matcher.is_match("notes.txt"));  // Matches *.TXT
```

## Pattern Syntax

### Glob Patterns

| Pattern | Matches | Example |
|---------|---------|---------|
| `*` | Any string (not `/`) | `*.rs` → `main.rs`, `lib.rs` |
| `**` | Any path segment | `**/*.py` → `a/b/c.py` |
| `?` | Single character | `?.txt` → `a.txt`, `1.txt` |
| `[abc]` | Character set | `[abc].rs` → `a.rs`, `b.rs` |
| `{a,b}` | Alternatives | `*.{rs,py}` → `main.rs`, `util.py` |
| `!pattern` | Negation | `!test*.py` → exclude test files |

### Gitignore Rules

| Pattern | Behavior |
|---------|----------|
| `pattern` | Matches in any directory |
| `/pattern` | Matches only at root |
| `dir/` | Matches directory only |
| `!pattern` | Negates previous patterns |
| `#comment` | Ignored line |

### Special Cases

- **Empty patterns**: Ignored (no effect)
- **Whitespace**: Leading/trailing whitespace is trimmed
- **Backslash escapes**: `\*` matches literal `*`
- **Unicode**: Full UTF-8 support for paths and patterns

## Performance

### Benchmarks

Pattern compilation and matching is highly optimized:

- **Glob compilation**: <1ms for typical pattern sets (10-50 patterns)
- **Path matching**: <1μs per path for compiled matchers
- **Literal matching**: <100ns using Aho-Corasick for large literal sets
- **Regex matching**: ~1-10μs depending on pattern complexity

### Optimizations

1. **Lazy compilation**: Patterns compiled only when first used
2. **Caching**: Compiled matchers cached in `OnceCell`
3. **Fast paths**: Literal string matching before expensive regex
4. **Set operations**: Boolean algebra simplification for pattern sets
5. **Aho-Corasick**: Multi-pattern matching for literals in O(n) time

## Configuration

### `MatchOptions`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `case_sensitive` | `bool` | Platform | Match case-sensitively |
| `require_literal_separator` | `bool` | `true` | `*` doesn't match `/` |
| `require_literal_leading_dot` | `bool` | `true` | `*` doesn't match `.hidden` |

### `IgnoreOptions`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hidden` | `bool` | `true` | Ignore hidden files (`.file`) |
| `parents` | `bool` | `true` | Check parent `.gitignore` files |
| `git_global` | `bool` | `false` | Use Git global ignore |
| `git_exclude` | `bool` | `false` | Use `.git/info/exclude` |

## Error Handling

All pattern operations return `Result<T, PatternError>`:

```rust
pub enum PatternError {
    InvalidGlob(String),      // Malformed glob syntax
    InvalidRegex(String),     // Malformed regex pattern
    IoError(io::Error),       // File read errors
    EmptyPatternSet,          // No patterns provided
}
```

## Integration

`scribe-patterns` is used throughout Scribe:

- **scribe-scanner**: Filters files during repository traversal
- **scribe-analysis**: Selects files for AST parsing
- **scribe-selection**: Applies include/exclude rules to selection
- **CLI**: Processes `--include` and `--exclude` flags

## See Also

- `scribe-scanner`: Repository scanning and filtering
- `scribe-selection`: File selection using patterns
- `scribe-core`: Shared types and configuration
- [globset documentation](https://docs.rs/globset): Underlying glob implementation

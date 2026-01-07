# Contributing

We welcome contributions to Scribe! This guide will help you get started.

## Getting Started

### Prerequisites

- Rust 1.75 or later
- Git
- A C compiler (for tree-sitter grammars)

### Clone and Build

```bash
git clone https://github.com/sibyllinesoft/scribe
cd scribe/scribe-rs
cargo build
```

### Run Tests

```bash
cargo test
```

### Run Locally

```bash
cargo run -- --help
cargo run -- --covering-set "src/main.rs" --stdout
```

## Project Structure

```
scribe/
├── scribe-rs/           # Rust workspace
│   ├── scribe-core/     # Shared types
│   ├── scribe-scanner/  # File scanning
│   ├── scribe-analysis/ # AST parsing
│   ├── scribe-graph/    # Dependency graphs
│   ├── scribe-selection/# Selection algorithms
│   ├── scribe-scaling/  # Token management
│   ├── scribe-webservice/ # HTTP API
│   └── src/             # CLI entry point
├── docs/                # Documentation (this site)
└── tests/               # Integration tests
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow the code style:

- Run `cargo fmt` before committing
- Run `cargo clippy` and fix warnings
- Add tests for new functionality

### 3. Test Your Changes

```bash
# Unit tests
cargo test

# Specific crate
cargo test -p scribe-selection

# Integration tests
cargo test --test integration
```

### 4. Submit a Pull Request

- Write a clear description of your changes
- Reference any related issues
- Ensure CI passes

## Code Style

### Rust Style

We follow standard Rust conventions:

```rust
// Good: descriptive names, proper documentation
/// Computes the covering set for a target entity.
///
/// # Arguments
/// * `target` - The entity to compute the covering set for
/// * `max_depth` - Maximum dependency traversal depth
///
/// # Returns
/// A `CoveringSetResult` containing the selected files
pub fn compute_covering_set(
    target: &CoveringSetTarget,
    max_depth: Option<usize>,
) -> Result<CoveringSetResult> {
    // Implementation
}
```

### Documentation

- All public items should have doc comments
- Include examples for complex functionality
- Keep the docs site updated

## Areas for Contribution

### Good First Issues

Look for issues labeled `good first issue`:

- Documentation improvements
- Test coverage
- Small bug fixes
- CLI usability improvements

### Larger Projects

- New language support (tree-sitter grammars)
- Additional selection algorithms
- Output format improvements
- Performance optimizations

## Adding Language Support

To add support for a new language:

1. **Add tree-sitter grammar** to `scribe-analysis/Cargo.toml`

2. **Implement parser** in `scribe-analysis/src/languages/`

```rust
pub struct MyLanguageParser;

impl LanguageParser for MyLanguageParser {
    fn parse(&self, content: &str) -> Result<ParseResult> {
        // Extract entities, imports, etc.
    }
}
```

3. **Add import resolution** in `scribe-graph/src/imports/`

```rust
pub fn resolve_mylang_imports(content: &str) -> Vec<Import> {
    // Parse import statements
}
```

4. **Add tests** with sample files

5. **Update documentation**

## Testing

### Unit Tests

Each module should have unit tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_covering_set_basic() {
        // Test implementation
    }
}
```

### Integration Tests

Add integration tests in `tests/`:

```rust
#[test]
fn test_full_workflow() {
    // Test end-to-end behavior
}
```

### Test Fixtures

Place test fixtures in `tests/fixtures/`:

```
tests/fixtures/
├── rust_project/
├── python_project/
└── typescript_project/
```

## Release Process

1. Update version in `Cargo.toml`
2. Update `CHANGELOG.md`
3. Create a release PR
4. After merge, tag the release
5. CI publishes to crates.io and npm

## Getting Help

- **Issues**: Open an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Pull Requests**: We review PRs promptly

## License

By contributing, you agree that your contributions will be licensed under the same terms as the project (MIT OR Apache-2.0).

---

Thank you for contributing to Scribe!

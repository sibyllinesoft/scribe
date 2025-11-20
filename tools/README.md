# Scribe Python Utilities

This directory contains Python support utilities for Scribe's operational scripts. While the core analysis engine is written in Rust, these lightweight Python helpers handle validation, security scanning, and other auxiliary tasks.

## Philosophy

Scribe keeps Python usage minimal and intentional:
- **Rust for core logic**: Repository scanning, AST analysis, graph algorithms, bundle generation
- **Python for scripting**: Validation utilities, security checks, and operational tools

This separation ensures the high-performance core remains in Rust while leveraging Python's ecosystem for scripting tasks.

## Structure

```
tools/
├── pyproject.toml              # Package configuration for scribe-support
└── scripts/
    ├── scan_secrets.py         # CLI wrapper for secret scanning
    ├── pack_verify.py          # CLI wrapper for bundle validation
    ├── ci_full_test.sh         # Shell script for CI testing
    └── support/                # Installable Python package
        ├── __init__.py         # Package exports
        ├── security.py         # SecretScanner implementation
        └── pack_validation.py  # PackVerifier implementation
```

## Installation

Install the support package in editable mode for development:

```bash
# From repository root
pip install -e tools

# This makes the following available:
from scripts.support import SecretScanner, PackVerifier
```

## Utilities

### SecretScanner (`scripts.support.SecretScanner`)

Scans directories for potential credentials and secrets using pattern matching.

**Features:**
- Detects API keys, tokens, passwords, and credential patterns
- Configurable severity levels (high, medium, low)
- JSON report generation with findings summary
- Integration with CI pipelines

**CLI Usage:**
```bash
# Scan current directory
python tools/scripts/scan_secrets.py

# Scan specific directory
python tools/scripts/scan_secrets.py --directory /path/to/repo

# Save report to JSON
python tools/scripts/scan_secrets.py --out security-report.json

# Verbose output with details
python tools/scripts/scan_secrets.py --verbose
```

**Python API:**
```python
from scripts.support import SecretScanner

scanner = SecretScanner()
findings = scanner.scan_directory(Path("/path/to/repo"))
report = scanner.summarize(findings)

# Report structure:
# {
#   "total_findings": 5,
#   "files_affected": 3,
#   "findings_by_severity": {"high": 2, "medium": 3},
#   "findings": [...]
# }
```

### PackVerifier (`scripts.support.PackVerifier`)

Validates Scribe bundle outputs against the JSON schema specification.

**Features:**
- Schema validation using `spec/index.schema.json`
- Token accounting verification
- Content hash integrity checks
- Metadata completeness validation

**CLI Usage:**
```bash
# Validate a bundle
python tools/scripts/pack_verify.py --validate bundle.json

# Write default schema to file
python tools/scripts/pack_verify.py --write-schema output.schema.json

# Use custom schema
python tools/scripts/pack_verify.py --schema custom.schema.json --validate bundle.json
```

**Python API:**
```python
from scripts.support import PackVerifier

verifier = PackVerifier(schema_path=Path("spec/index.schema.json"))
result = verifier.validate_pack(bundle_data)

if result["is_valid"]:
    print("Bundle is valid!")
else:
    print("Validation errors:")
    for error in result["errors"]:
        print(f"  - {error}")
```

## CI Integration

The utilities are designed for CI/CD pipeline integration:

**GitHub Actions Example:**
```yaml
- name: Scan for secrets
  run: |
    pip install -e tools
    python tools/scripts/scan_secrets.py --directory . --verbose

- name: Validate generated bundle
  run: |
    pip install -e tools
    python tools/scripts/pack_verify.py --validate output/bundle.json
```

**Script:** `tools/scripts/ci_full_test.sh` runs comprehensive checks including:
- Secret scanning across repository
- Bundle validation for test outputs
- Python linting and type checking (if configured)

## Development

### Adding New Utilities

To add a new utility:

1. **Create implementation** in `support/`:
   ```python
   # tools/scripts/support/my_utility.py
   class MyUtility:
       def process(self, data):
           # Implementation
           pass
   ```

2. **Export from package**:
   ```python
   # tools/scripts/support/__init__.py
   from .my_utility import MyUtility
   __all__ = ["SecretScanner", "PackVerifier", "MyUtility"]
   ```

3. **Create CLI wrapper**:
   ```python
   # tools/scripts/my_tool.py
   from scripts.support import MyUtility
   # CLI implementation
   ```

4. **Add tests** in `tests/` directory

### Dependencies

Keep dependencies minimal (currently only `jsonschema>=4.0`). Add new dependencies to `pyproject.toml`:

```toml
[project]
dependencies = [
  "jsonschema>=4.0",
  "new-package>=1.0",
]
```

## Testing

```bash
# Install with dev dependencies
pip install -e tools[dev]

# Run tests (if test suite exists)
pytest tools/tests/

# Type checking
mypy tools/scripts/support/

# Linting
ruff check tools/scripts/
```

## Design Principles

1. **Minimal coupling**: Python utilities don't depend on Rust internals
2. **Simple interfaces**: Clear, documented APIs for common tasks
3. **CI-friendly**: Exit codes, JSON reports, verbose modes for automation
4. **Installable**: Proper package structure for `pip install -e`
5. **Self-contained**: Each utility can run independently

## See Also

- `spec/README.md`: JSON schema specification for bundle format
- `ARCHITECTURE.md`: Overall system design
- `scribe-rs/`: Rust workspace with core analysis engine

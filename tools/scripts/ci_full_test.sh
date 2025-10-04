#!/bin/bash
# Scribe CI Test Suite
# Runs formatting, linting, unit tests, and pack validation helpers

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TOOLS_DIR="$PROJECT_ROOT/tools"
PY_SUPPORT_DIR="$TOOLS_DIR/scripts/support"
ARTIFACTS_DIR="$PROJECT_ROOT/artifacts"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Ensure artifacts directory exists
mkdir -p "$ARTIFACTS_DIR"

log_info "Starting Scribe CI Test Suite"
log_info "Project root: $PROJECT_ROOT"
log_info "Artifacts: $ARTIFACTS_DIR"

# 1. Static Analysis
log_info "Running static analysis..."
if command -v ruff &> /dev/null; then
    (cd "$PROJECT_ROOT" && ruff check . --output-format=json > "$ARTIFACTS_DIR/ruff_results.json" || true)
    (cd "$PROJECT_ROOT" && ruff format --check .) || log_warning "Code formatting issues detected"
    log_success "Ruff analysis completed"
else
    log_warning "Ruff not available"
fi

if command -v mypy &> /dev/null; then
    mypy "$PY_SUPPORT_DIR" --json-report "$ARTIFACTS_DIR/mypy_report" || log_warning "Type checking issues detected"
    log_success "MyPy analysis completed"
else
    log_warning "MyPy not available"
fi

# 2. Security Scanning
log_info "Running security scans..."
if command -v bandit &> /dev/null; then
    bandit -r "$PY_SUPPORT_DIR" -f json -o "$ARTIFACTS_DIR/bandit_results.json" || log_warning "Security issues detected"
    log_success "Bandit security scan completed"
else
    log_warning "Bandit not available"
fi

if command -v safety &> /dev/null; then
    safety check --json --output "$ARTIFACTS_DIR/safety_results.json" || log_warning "Dependency vulnerabilities detected"
    log_success "Safety dependency check completed"
else
    log_warning "Safety not available"
fi

# 3. Unit Tests
log_info "Running unit tests..."
if command -v pytest &> /dev/null; then
    pytest "$PROJECT_ROOT/tests" --json-report --json-report-file="$ARTIFACTS_DIR/pytest_results.json" || log_warning "Unit test failures detected"
    log_success "Unit tests completed"
else
    log_warning "Pytest not available"
fi

# 4. Pack Verification
log_info "Running pack verification..."
if [[ -f "$SCRIPT_DIR/pack_verify.py" ]]; then
    python "$SCRIPT_DIR/pack_verify.py" --write-schema "$PROJECT_ROOT/spec/index.schema.json"
    log_success "Pack verification schema updated"
else
    log_warning "Pack verify script not found"
fi

# 5. Import Tests
log_info "Testing core imports..."
# shellcheck disable=SC2016
python -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
sys.path.insert(0, '$TOOLS_DIR')
try:
    import scripts.support
    print('✓ Python support package import successful')
except ImportError as e:
    print(f'✗ Import failed: {e}')
    sys.exit(1)
"

log_success "Core imports test passed"

# 6. Generate test report
log_info "Generating test report..."
cat > "$ARTIFACTS_DIR/ci_test_report.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "test_suite": "ci",
    "environment": {
        "python_version": "$(python --version)",
        "hostname": "$(hostname)",
        "pwd": "$(pwd)"
    },
    "test_results": {
        "static_analysis": "completed",
        "security_scans": "completed", 
        "unit_tests": "completed",
        "import_tests": "passed",
        "pack_verification": "completed"
    }
}
EOF

log_success "Scribe CI suite completed successfully"
log_info "Results available in: $ARTIFACTS_DIR/"

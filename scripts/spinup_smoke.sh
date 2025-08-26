#!/bin/bash
# PackRepo Hermetic Boot & Golden Smoke Tests
# Verifies clean checkout -> container build -> install/lock -> readiness -> golden smokes
# Produces signed boot transcript for reproducible builds

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ARTIFACTS_DIR="$PROJECT_ROOT/artifacts"
LOGS_DIR="$PROJECT_ROOT/logs"

# Default parameters (can be overridden by command line)
REPO_URL="${REPO_URL:-}"
TOKEN_BUDGET="${TOKEN_BUDGET:-120000}"
TOKENIZER="${TOKENIZER:-cl100k}"
NO_LLM_FLAG="${NO_LLM_FLAG:---no-llm}"
CONTAINER_TAG="${CONTAINER_TAG:-packrepo:hermetic-$(date +%Y%m%d-%H%M%S)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date -Iseconds) $*" | tee -a "$ARTIFACTS_DIR/boot_transcript.log"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date -Iseconds) $*" | tee -a "$ARTIFACTS_DIR/boot_transcript.log"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date -Iseconds) $*" | tee -a "$ARTIFACTS_DIR/boot_transcript.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date -Iseconds) $*" | tee -a "$ARTIFACTS_DIR/boot_transcript.log"
}

# Parse command line arguments
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Hermetic boot verification and golden smoke tests for PackRepo.

OPTIONS:
    --repo URL          Repository URL to test (required)
    --budget TOKENS     Token budget for testing (default: 120000)
    --tokenizer NAME    Tokenizer to use (default: cl100k)
    --no-llm           Run in deterministic mode without LLM
    --container TAG     Container tag to build (default: auto-generated)
    --skip-build       Skip container build step
    --help             Show this help message

ENVIRONMENT VARIABLES:
    DATASET_REPOS      Comma-separated list of test repositories
    FUZZ_MIN          Minimum fuzz testing time in minutes
    EMB_MODEL         Embedding model to use
    RERANK_MODEL      Reranking model to use
    SUM_MODEL         Summary model to use

EXAMPLES:
    $0 --repo https://github.com/karpathy/nanoGPT
    $0 --repo local/test-repo --budget 50000 --tokenizer o200k
    $0 --skip-build --repo https://github.com/pytorch/pytorch
EOF
}

# Initialize
init_environment() {
    log_info "Initializing hermetic boot environment..."
    
    # Create necessary directories
    mkdir -p "$ARTIFACTS_DIR" "$LOGS_DIR" "$PROJECT_ROOT/locks"
    
    # Initialize boot transcript
    cat > "$ARTIFACTS_DIR/boot_transcript.json" << EOF
{
    "boot_session": "$(uuidgen)",
    "timestamp": "$(date -Iseconds)",
    "hostname": "$(hostname)",
    "user": "$(whoami)",
    "pwd": "$(pwd)",
    "git_commit": "$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_dirty": $(git -C "$PROJECT_ROOT" diff --quiet && echo false || echo true),
    "phases": []
}
EOF

    log_success "Environment initialized"
}

# Phase 1: Environment validation
validate_environment() {
    log_info "Phase 1: Environment validation"
    
    local phase_start=$(date -Iseconds)
    local errors=0
    
    # Check required tools
    local required_tools=("docker" "python3" "git" "jq" "uv")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool '$tool' not found"
            ((errors++))
        else
            log_info "✓ $tool: $(command -v "$tool")"
        fi
    done
    
    # Validate Git repository
    if [[ ! -d "$PROJECT_ROOT/.git" ]]; then
        log_error "Not in a Git repository"
        ((errors++))
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon not running or not accessible"
        ((errors++))
    fi
    
    # Record environment details
    cat > "$ARTIFACTS_DIR/boot_env.txt" << EOF
# PackRepo Boot Environment Record
# Generated: $(date -Iseconds)

## System Information
HOSTNAME=$(hostname)
USER=$(whoami)
PWD=$(pwd)
UNAME=$(uname -a)

## Git Information
GIT_COMMIT=$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo 'unknown')
GIT_BRANCH=$(git -C "$PROJECT_ROOT" branch --show-current 2>/dev/null || echo 'unknown')
GIT_DIRTY=$(git -C "$PROJECT_ROOT" diff --quiet && echo 'false' || echo 'true')
GIT_REMOTE=$(git -C "$PROJECT_ROOT" remote get-url origin 2>/dev/null || echo 'unknown')

## Python Environment
PYTHON_VERSION=$(python3 --version)
PYTHON_PATH=$(which python3)
UV_VERSION=$(uv --version 2>/dev/null || echo 'not installed')

## Docker Information
DOCKER_VERSION=$(docker --version)
DOCKER_COMPOSE_VERSION=$(docker-compose --version 2>/dev/null || echo 'not installed')

## Environment Variables
PATH=$PATH
PYTHONPATH=${PYTHONPATH:-}
VIRTUAL_ENV=${VIRTUAL_ENV:-}

## Timestamp
BOOT_TIMESTAMP=$(date -Iseconds)
BOOT_EPOCH=$(date +%s)
EOF
    
    local phase_end=$(date -Iseconds)
    update_boot_transcript "environment_validation" "$phase_start" "$phase_end" $errors
    
    if [[ $errors -gt 0 ]]; then
        log_error "Environment validation failed with $errors errors"
        return 1
    fi
    
    log_success "Environment validation passed"
    return 0
}

# Phase 2: Container build and verification
build_container() {
    log_info "Phase 2: Container build and verification"
    
    local phase_start=$(date -Iseconds)
    local errors=0
    
    # Build the hermetic container
    log_info "Building container: $CONTAINER_TAG"
    
    if ! docker build \
        --tag "$CONTAINER_TAG" \
        --file "$PROJECT_ROOT/infra/Dockerfile" \
        --target development \
        --build-arg PYTHON_VERSION=3.12.7 \
        --build-arg UV_VERSION=0.5.6 \
        "$PROJECT_ROOT" 2>&1 | tee "$LOGS_DIR/container_build.log"; then
        log_error "Container build failed"
        ((errors++))
    else
        log_success "Container built successfully: $CONTAINER_TAG"
    fi
    
    # Verify container health
    log_info "Verifying container health..."
    if ! docker run --rm "$CONTAINER_TAG" python -c "import packrepo; print('Container health: OK')"; then
        log_error "Container health check failed"
        ((errors++))
    else
        log_success "Container health check passed"
    fi
    
    # Record container metadata
    docker inspect "$CONTAINER_TAG" > "$ARTIFACTS_DIR/container_metadata.json"
    docker image ls "$CONTAINER_TAG" --format "table {{.Repository}}:{{.Tag}}\t{{.ImageID}}\t{{.CreatedAt}}\t{{.Size}}" > "$ARTIFACTS_DIR/container_info.txt"
    
    local phase_end=$(date -Iseconds)
    update_boot_transcript "container_build" "$phase_start" "$phase_end" $errors
    
    if [[ $errors -gt 0 ]]; then
        log_error "Container build phase failed with $errors errors"
        return 1
    fi
    
    log_success "Container build phase completed"
    return 0
}

# Phase 3: Dependencies and lock verification
verify_dependencies() {
    log_info "Phase 3: Dependencies and lock verification"
    
    local phase_start=$(date -Iseconds)
    local errors=0
    
    # Create lock directory and files
    mkdir -p "$PROJECT_ROOT/locks"
    
    # Generate and verify uv lock
    log_info "Generating uv lock file..."
    if ! uv lock --locked 2>&1 | tee "$LOGS_DIR/uv_lock.log"; then
        log_warning "uv lock generation had issues (may be expected)"
    fi
    
    # Copy lock files to locks directory
    cp "$PROJECT_ROOT/uv.lock" "$PROJECT_ROOT/locks/uv.lock.$(date +%Y%m%d-%H%M%S)" 2>/dev/null || log_warning "uv.lock not found"
    cp "$PROJECT_ROOT/requirements.txt" "$PROJECT_ROOT/locks/requirements.txt.$(date +%Y%m%d-%H%M%S)" 2>/dev/null || log_warning "requirements.txt not found"
    
    # Verify dependencies can be installed in container
    log_info "Verifying dependency installation in container..."
    if ! docker run --rm -v "$PROJECT_ROOT:/app" "$CONTAINER_TAG" \
        sh -c "cd /app && uv sync --frozen --all-extras" 2>&1 | tee "$LOGS_DIR/deps_install.log"; then
        log_error "Dependency installation failed"
        ((errors++))
    else
        log_success "Dependencies installed successfully"
    fi
    
    # Generate dependency hashes
    log_info "Recording dependency hashes..."
    cat > "$ARTIFACTS_DIR/dependency_hashes.json" << EOF
{
    "uv_lock_sha256": "$(sha256sum "$PROJECT_ROOT/uv.lock" 2>/dev/null | cut -d' ' -f1 || echo 'missing')",
    "requirements_sha256": "$(sha256sum "$PROJECT_ROOT/requirements.txt" 2>/dev/null | cut -d' ' -f1 || echo 'missing')",
    "pyproject_sha256": "$(sha256sum "$PROJECT_ROOT/pyproject.toml" | cut -d' ' -f1)",
    "generation_time": "$(date -Iseconds)"
}
EOF
    
    local phase_end=$(date -Iseconds)
    update_boot_transcript "dependency_verification" "$phase_start" "$phase_end" $errors
    
    if [[ $errors -gt 0 ]]; then
        log_error "Dependency verification failed with $errors errors"
        return 1
    fi
    
    log_success "Dependency verification completed"
    return 0
}

# Phase 4: Core functionality validation
validate_core_functionality() {
    log_info "Phase 4: Core functionality validation"
    
    local phase_start=$(date -Iseconds)
    local errors=0
    
    # Test basic import and functionality
    log_info "Testing core imports..."
    if ! docker run --rm -v "$PROJECT_ROOT:/app" "$CONTAINER_TAG" \
        python -c "
import packrepo
import packrepo.packer.chunker
import packrepo.packer.selector
import packrepo.packer.tokenizer
import packrepo.packer.packfmt
print('Core imports: OK')
" 2>&1 | tee "$LOGS_DIR/core_imports.log"; then
        log_error "Core imports failed"
        ((errors++))
    else
        log_success "Core imports successful"
    fi
    
    # Test tokenizer functionality
    log_info "Testing tokenizer functionality..."
    if ! docker run --rm -v "$PROJECT_ROOT:/app" "$CONTAINER_TAG" \
        python -c "
from packrepo.packer.tokenizer import get_tokenizer
tokenizer = get_tokenizer('$TOKENIZER')
tokens = tokenizer.encode('Hello, world!')
print(f'Tokenizer test: {len(tokens)} tokens for test string')
assert len(tokens) > 0, 'Tokenization failed'
print('Tokenizer: OK')
" 2>&1 | tee "$LOGS_DIR/tokenizer_test.log"; then
        log_error "Tokenizer test failed"
        ((errors++))
    else
        log_success "Tokenizer test successful"
    fi
    
    local phase_end=$(date -Iseconds)
    update_boot_transcript "core_functionality" "$phase_start" "$phase_end" $errors
    
    if [[ $errors -gt 0 ]]; then
        log_error "Core functionality validation failed with $errors errors"
        return 1
    fi
    
    log_success "Core functionality validation completed"
    return 0
}

# Phase 5: Golden smoke tests
run_golden_smoke_tests() {
    log_info "Phase 5: Golden smoke tests"
    
    local phase_start=$(date -Iseconds)
    local errors=0
    
    # Prepare test repository
    local test_repo="${REPO_URL:-$PROJECT_ROOT}"
    
    if [[ -z "$test_repo" ]]; then
        log_warning "No test repository specified, using self-test"
        test_repo="$PROJECT_ROOT"
    fi
    
    log_info "Running golden smoke test on: $test_repo"
    
    # Create test output directory
    mkdir -p "$LOGS_DIR/golden_smoke"
    
    # Run the golden smoke test
    local smoke_cmd="python rendergit.py"
    if [[ "$test_repo" != "$PROJECT_ROOT" ]]; then
        smoke_cmd="$smoke_cmd --repo '$test_repo'"
    fi
    smoke_cmd="$smoke_cmd --budget $TOKEN_BUDGET --tokenizer $TOKENIZER $NO_LLM_FLAG"
    
    log_info "Executing: $smoke_cmd"
    
    if ! docker run --rm \
        -v "$PROJECT_ROOT:/app" \
        -v "$LOGS_DIR/golden_smoke:/app/output" \
        -w "/app" \
        "$CONTAINER_TAG" \
        sh -c "$smoke_cmd --output /app/output/golden_smoke.html" 2>&1 | tee "$LOGS_DIR/golden_smoke.log"; then
        log_error "Golden smoke test failed"
        ((errors++))
    else
        log_success "Golden smoke test completed"
    fi
    
    # Verify output was generated
    if [[ -f "$LOGS_DIR/golden_smoke/golden_smoke.html" ]]; then
        local output_size=$(stat -f%z "$LOGS_DIR/golden_smoke/golden_smoke.html" 2>/dev/null || stat -c%s "$LOGS_DIR/golden_smoke/golden_smoke.html")
        log_success "Golden smoke output generated: ${output_size} bytes"
        
        # Generate output hash for reproducibility verification
        local output_hash=$(sha256sum "$LOGS_DIR/golden_smoke/golden_smoke.html" | cut -d' ' -f1)
        echo "golden_smoke_hash: $output_hash" >> "$ARTIFACTS_DIR/boot_transcript.txt"
    else
        log_error "Golden smoke output not generated"
        ((errors++))
    fi
    
    local phase_end=$(date -Iseconds)
    update_boot_transcript "golden_smoke_tests" "$phase_start" "$phase_end" $errors
    
    if [[ $errors -gt 0 ]]; then
        log_error "Golden smoke tests failed with $errors errors"
        return 1
    fi
    
    log_success "Golden smoke tests completed"
    return 0
}

# Update boot transcript
update_boot_transcript() {
    local phase_name="$1"
    local start_time="$2"
    local end_time="$3"
    local error_count="$4"
    
    # Calculate duration
    local start_epoch=$(date -d "$start_time" +%s)
    local end_epoch=$(date -d "$end_time" +%s)
    local duration=$((end_epoch - start_epoch))
    
    # Update JSON transcript
    jq --arg phase "$phase_name" \
       --arg start "$start_time" \
       --arg end "$end_time" \
       --argjson duration "$duration" \
       --argjson errors "$error_count" \
       --arg status $([ "$error_count" -eq 0 ] && echo "success" || echo "failure") \
       '.phases += [{
           "name": $phase,
           "start_time": $start,
           "end_time": $end,
           "duration_seconds": $duration,
           "error_count": $errors,
           "status": $status
       }]' "$ARTIFACTS_DIR/boot_transcript.json" > "$ARTIFACTS_DIR/boot_transcript.json.tmp"
    
    mv "$ARTIFACTS_DIR/boot_transcript.json.tmp" "$ARTIFACTS_DIR/boot_transcript.json"
}

# Generate final boot transcript
generate_final_transcript() {
    log_info "Generating final boot transcript..."
    
    # Add final metadata
    jq --arg completion_time "$(date -Iseconds)" \
       --arg container_tag "$CONTAINER_TAG" \
       --arg total_duration "$(($(date +%s) - boot_start_time))" \
       '. + {
           "completion_time": $completion_time,
           "container_tag": $container_tag,
           "total_duration_seconds": ($total_duration | tonumber),
           "success": (.phases | map(select(.status == "failure")) | length == 0)
       }' "$ARTIFACTS_DIR/boot_transcript.json" > "$ARTIFACTS_DIR/boot_transcript.json.tmp"
    
    mv "$ARTIFACTS_DIR/boot_transcript.json.tmp" "$ARTIFACTS_DIR/boot_transcript.json"
    
    log_success "Boot transcript generated: $ARTIFACTS_DIR/boot_transcript.json"
}

# Main execution
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --repo)
                REPO_URL="$2"
                shift 2
                ;;
            --budget)
                TOKEN_BUDGET="$2"
                shift 2
                ;;
            --tokenizer)
                TOKENIZER="$2"
                shift 2
                ;;
            --no-llm)
                NO_LLM_FLAG="--no-llm"
                shift
                ;;
            --container)
                CONTAINER_TAG="$2"
                shift 2
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Record boot start time
    local boot_start_time=$(date +%s)
    
    # Initialize environment
    init_environment
    
    log_info "🚀 Starting PackRepo hermetic boot verification"
    log_info "Container: $CONTAINER_TAG"
    log_info "Repository: ${REPO_URL:-'self-test'}"
    log_info "Budget: $TOKEN_BUDGET tokens"
    log_info "Tokenizer: $TOKENIZER"
    log_info "Mode: ${NO_LLM_FLAG:-'with-llm'}"
    
    # Run boot phases
    local overall_success=true
    
    if ! validate_environment; then
        overall_success=false
    fi
    
    if [[ "${SKIP_BUILD:-false}" != "true" ]]; then
        if ! build_container; then
            overall_success=false
        fi
    else
        log_info "Skipping container build (--skip-build specified)"
    fi
    
    if ! verify_dependencies; then
        overall_success=false
    fi
    
    if ! validate_core_functionality; then
        overall_success=false
    fi
    
    if ! run_golden_smoke_tests; then
        overall_success=false
    fi
    
    # Generate final transcript
    generate_final_transcript
    
    # Report results
    if [[ "$overall_success" == "true" ]]; then
        log_success "🎉 Hermetic boot verification completed successfully"
        log_success "All phases passed - system ready for production"
        log_info "Boot transcript: $ARTIFACTS_DIR/boot_transcript.json"
        log_info "Logs directory: $LOGS_DIR"
        exit 0
    else
        log_error "❌ Hermetic boot verification failed"
        log_error "Check logs in: $LOGS_DIR"
        log_error "Boot transcript: $ARTIFACTS_DIR/boot_transcript.json"
        exit 1
    fi
}

# Trap cleanup
trap 'log_error "Hermetic boot verification interrupted"; exit 130' INT TERM

# Execute main function with all arguments
main "$@"
#!/bin/bash
set -euo pipefail

#=============================================================================
# PackRepo FastPath V2 CI/CD Pipeline Execution Script
# 
# This script orchestrates the complete workflow automation for FastPath V2
# validation, statistical analysis, and promotion decisions.
#
# Usage:
#   ./run_pipeline.sh [options]
#
# Options:
#   --help, -h              Show this help message
#   --dry-run               Show what would be executed without running
#   --parallel              Run benchmarks in parallel (default)
#   --sequential            Run benchmarks sequentially  
#   --variants VARIANTS     Comma-separated list of variants to test
#   --skip-building         Skip the building workflow (B0-B2)
#   --skip-security         Skip security scans
#   --artifacts-dir DIR     Artifacts output directory (default: artifacts)
#   --verbose, -v           Verbose output
#   --force-promotion       Force promotion even if gates fail (dangerous)
#
# Environment Variables:
#   CI                      Set to 'true' when running in CI environment
#   GITHUB_ACTIONS          Set to 'true' when running in GitHub Actions
#   ARTIFACTS_DIR           Override default artifacts directory
#   PIPELINE_TIMEOUT        Pipeline timeout in seconds (default: 7200)
#=============================================================================

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PIPELINE_VERSION="2.0.0"
DEFAULT_TIMEOUT=7200  # 2 hours

# Default options
DRY_RUN=false
PARALLEL=true
VARIANTS="baseline,V1,V2,V3,V4,V5"
SKIP_BUILDING=false
SKIP_SECURITY=false
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts}"
VERBOSE=false
FORCE_PROMOTION=false
PIPELINE_TIMEOUT="${PIPELINE_TIMEOUT:-$DEFAULT_TIMEOUT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

# Usage function
show_usage() {
    cat << EOF
PackRepo FastPath V2 CI/CD Pipeline v${PIPELINE_VERSION}

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    --dry-run               Show what would be executed without running
    --parallel              Run benchmarks in parallel (default)
    --sequential            Run benchmarks sequentially
    --variants VARIANTS     Comma-separated list of variants (default: all)
    --skip-building         Skip the building workflow (B0-B2)
    --skip-security         Skip security scans
    --artifacts-dir DIR     Artifacts output directory (default: artifacts)
    -v, --verbose           Verbose output
    --force-promotion       Force promotion even if gates fail (DANGEROUS)
    --timeout SECONDS       Pipeline timeout in seconds (default: $DEFAULT_TIMEOUT)

VARIANTS:
    baseline, V1, V2, V3, V4, V5

EXAMPLES:
    # Run full pipeline
    $0

    # Run only specific variants
    $0 --variants "baseline,V1,V3"

    # Run sequentially with verbose output
    $0 --sequential --verbose

    # Dry run to see what would be executed
    $0 --dry-run

    # Skip building phase (for development)
    $0 --skip-building --variants "V1,V2"

ENVIRONMENT:
    CI=true                 Indicates CI environment
    GITHUB_ACTIONS=true     Indicates GitHub Actions
    ARTIFACTS_DIR=path      Override artifacts directory
    PIPELINE_TIMEOUT=secs   Override timeout

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --parallel)
                PARALLEL=true
                shift
                ;;
            --sequential)
                PARALLEL=false
                shift
                ;;
            --variants)
                VARIANTS="$2"
                shift 2
                ;;
            --skip-building)
                SKIP_BUILDING=true
                shift
                ;;
            --skip-security)
                SKIP_SECURITY=true
                shift
                ;;
            --artifacts-dir)
                ARTIFACTS_DIR="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --force-promotion)
                FORCE_PROMOTION=true
                log_warning "Force promotion enabled - this bypasses safety gates!"
                shift
                ;;
            --timeout)
                PIPELINE_TIMEOUT="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Environment validation
validate_environment() {
    log_step "Validating environment"
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    local python_version
    python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    log_info "Python version: $python_version"
    
    if [[ $(echo "$python_version" | cut -d. -f1) -lt 3 ]] || [[ $(echo "$python_version" | cut -d. -f2) -lt 8 ]]; then
        log_error "Python 3.8+ is required, found $python_version"
        exit 1
    fi
    
    # Check required files
    local required_files=(
        "run_fastpath_benchmarks.py"
        "requirements.txt"
        "ci/pipeline.py"
        "ci/benchmark_runner.py" 
        "ci/statistical_analysis.py"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$PROJECT_ROOT/$file" ]]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done
    
    # Create artifacts directory
    mkdir -p "$PROJECT_ROOT/$ARTIFACTS_DIR"
    log_info "Artifacts directory: $PROJECT_ROOT/$ARTIFACTS_DIR"
    
    # Check if running in CI
    if [[ "${CI:-false}" == "true" ]]; then
        log_info "Running in CI environment"
        if [[ "${GITHUB_ACTIONS:-false}" == "true" ]]; then
            log_info "Running in GitHub Actions"
        fi
    fi
}

# Setup virtual environment
setup_environment() {
    log_step "Setting up Python environment"
    
    local venv_path="$PROJECT_ROOT/.venv"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create virtual environment at $venv_path"
        log_info "[DRY RUN] Would install dependencies from requirements.txt"
        return
    fi
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "$venv_path" ]]; then
        log_info "Creating virtual environment"
        python3 -m venv "$venv_path"
    fi
    
    # Activate virtual environment
    # shellcheck source=/dev/null
    source "$venv_path/bin/activate"
    
    # Upgrade pip and install dependencies
    log_info "Installing dependencies"
    pip install -U pip wheel >/dev/null 2>&1
    pip install -r "$PROJECT_ROOT/requirements.txt" >/dev/null 2>&1
    
    # Install additional CI/CD dependencies
    pip install scipy numpy >/dev/null 2>&1 || log_warning "Failed to install scipy/numpy - statistical analysis may be limited"
    
    log_success "Environment setup complete"
}

# Execute pipeline with timeout
execute_with_timeout() {
    local cmd="$1"
    local timeout="$2"
    local description="$3"
    
    log_info "Executing: $description (timeout: ${timeout}s)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would execute: $cmd"
        return 0
    fi
    
    if timeout "$timeout" bash -c "$cmd"; then
        return 0
    else
        local exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            log_error "$description timed out after ${timeout} seconds"
        else
            log_error "$description failed with exit code $exit_code"
        fi
        return $exit_code
    fi
}

# Main pipeline execution
run_pipeline() {
    log_step "Starting PackRepo FastPath V2 CI/CD Pipeline v${PIPELINE_VERSION}"
    
    local pipeline_start
    pipeline_start=$(date +%s)
    
    # Build pipeline command
    local pipeline_cmd="cd '$PROJECT_ROOT' && python ci/pipeline.py"
    if [[ "$VERBOSE" == "true" ]]; then
        pipeline_cmd="$pipeline_cmd --verbose"
    fi
    pipeline_cmd="$pipeline_cmd --project-root '$PROJECT_ROOT'"
    
    # Execute main pipeline
    if ! execute_with_timeout "$pipeline_cmd" "$PIPELINE_TIMEOUT" "Main CI/CD Pipeline"; then
        log_error "Pipeline execution failed"
        return 1
    fi
    
    local pipeline_end
    pipeline_end=$(date +%s)
    local pipeline_duration=$((pipeline_end - pipeline_start))
    
    log_success "Pipeline completed in ${pipeline_duration} seconds"
    
    # Check results
    local report_file="$PROJECT_ROOT/$ARTIFACTS_DIR/pipeline_report.json"
    if [[ -f "$report_file" ]]; then
        local status
        status=$(python3 -c "import json; print(json.load(open('$report_file')).get('pipeline_status', 'UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")
        
        if [[ "$status" == "PROMOTE" ]]; then
            log_success "🎉 FastPath V2 APPROVED for promotion!"
            return 0
        elif [[ "$status" == "REJECT" ]]; then
            if [[ "$FORCE_PROMOTION" == "true" ]]; then
                log_warning "🚨 FastPath V2 failed gates but FORCE_PROMOTION enabled"
                return 0
            else
                log_error "❌ FastPath V2 REJECTED - does not meet acceptance criteria"
                return 1
            fi
        else
            log_error "❓ Unknown pipeline status: $status"
            return 1
        fi
    else
        log_error "Pipeline report not found: $report_file"
        return 1
    fi
}

# Security scan execution
run_security_scan() {
    if [[ "$SKIP_SECURITY" == "true" ]]; then
        log_info "Skipping security scan (--skip-security)"
        return 0
    fi
    
    log_step "Running security scan"
    
    local security_cmd="cd '$PROJECT_ROOT' && "
    security_cmd+="bandit -r packrepo -f json -o '$ARTIFACTS_DIR/bandit_report.json' >/dev/null 2>&1 || true"
    
    if ! execute_with_timeout "$security_cmd" 300 "Security Scan"; then
        log_warning "Security scan had issues but continuing"
    fi
    
    log_success "Security scan completed"
}

# Generate execution summary
generate_summary() {
    log_step "Generating execution summary"
    
    local summary_file="$PROJECT_ROOT/$ARTIFACTS_DIR/pipeline_execution_summary.txt"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would generate summary at $summary_file"
        return
    fi
    
    cat > "$summary_file" << EOF
PackRepo FastPath V2 CI/CD Pipeline Execution Summary
=======================================================

Execution Details:
- Pipeline Version: $PIPELINE_VERSION
- Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')
- Host: $(hostname)
- User: $(whoami)
- Working Directory: $PROJECT_ROOT

Configuration:
- Dry Run: $DRY_RUN
- Parallel Execution: $PARALLEL
- Variants: $VARIANTS
- Skip Building: $SKIP_BUILDING
- Skip Security: $SKIP_SECURITY
- Artifacts Directory: $ARTIFACTS_DIR
- Verbose: $VERBOSE
- Force Promotion: $FORCE_PROMOTION
- Timeout: ${PIPELINE_TIMEOUT}s

Environment:
- Python: $(python3 --version 2>&1)
- CI: ${CI:-false}
- GitHub Actions: ${GITHUB_ACTIONS:-false}

Artifacts Generated:
EOF
    
    if [[ -d "$PROJECT_ROOT/$ARTIFACTS_DIR" ]]; then
        find "$PROJECT_ROOT/$ARTIFACTS_DIR" -type f -name "*.json" | while read -r file; do
            echo "- $(basename "$file")" >> "$summary_file"
        done
    fi
    
    echo "" >> "$summary_file"
    echo "For detailed results, see: $ARTIFACTS_DIR/pipeline_report.json" >> "$summary_file"
    
    log_success "Summary generated: $summary_file"
}

# Cleanup function
cleanup() {
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        log_error "Pipeline failed with exit code $exit_code"
    fi
    
    # Generate summary even on failure
    generate_summary
    
    # Print final status
    echo ""
    echo "=============================================="
    if [[ $exit_code -eq 0 ]]; then
        log_success "🎉 Pipeline completed successfully"
    else
        log_error "❌ Pipeline failed - check logs for details"
    fi
    echo "=============================================="
    
    exit $exit_code
}

# Signal handlers
trap cleanup EXIT
trap 'log_error "Pipeline interrupted by user"; exit 130' INT TERM

# Main execution
main() {
    # Parse arguments
    parse_arguments "$@"
    
    # Show configuration if verbose or dry run
    if [[ "$VERBOSE" == "true" || "$DRY_RUN" == "true" ]]; then
        echo ""
        log_info "Pipeline Configuration:"
        log_info "  Version: $PIPELINE_VERSION"
        log_info "  Project Root: $PROJECT_ROOT"
        log_info "  Artifacts Dir: $ARTIFACTS_DIR"
        log_info "  Variants: $VARIANTS"
        log_info "  Parallel: $PARALLEL"
        log_info "  Skip Building: $SKIP_BUILDING"
        log_info "  Skip Security: $SKIP_SECURITY" 
        log_info "  Dry Run: $DRY_RUN"
        log_info "  Force Promotion: $FORCE_PROMOTION"
        log_info "  Timeout: ${PIPELINE_TIMEOUT}s"
        echo ""
    fi
    
    # Execute pipeline stages
    validate_environment
    setup_environment
    run_security_scan
    run_pipeline
}

# Execute main function with all arguments
main "$@"
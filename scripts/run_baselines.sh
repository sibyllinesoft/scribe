#!/bin/bash
# FastPath V5 Baseline Runner - Execute V1-V4 baselines with complete reproducibility
# Implements TODO.md V1-V4 Run Matrix requirements

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_ROOT}/results/baselines"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${RESULTS_DIR}/baseline_execution_${TIMESTAMP}.log"

# Default parameters
REPO_PATH="${REPO_PATH:-$PROJECT_ROOT}"
TOKEN_BUDGET="${TOKEN_BUDGET:-120000}"
RANDOM_SEED="${RANDOM_SEED:-42}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-json}"

# Baseline configurations - matching TODO.md requirements
declare -A BASELINES
BASELINES[V1]="random,Random baseline,±5% runtime,Anchor floor"
BASELINES[V2]="recency,Recency baseline,±5% runtime,Show temporal bias"  
BASELINES[V3]="tfidf,TF-IDF baseline,±5% runtime,Anchor mid"
BASELINES[V4]="semantic,Semantic baseline,±5% runtime,Anchor high"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Usage information
show_usage() {
    cat << EOF
FastPath V5 Baseline Runner

Usage: $0 [OPTIONS] [BASELINE_IDS...]

OPTIONS:
    --random        Run V1 random baseline only
    --recency       Run V2 recency baseline only  
    --tfidf         Run V3 TF-IDF baseline only
    --semantic      Run V4 semantic baseline only
    --all           Run all V1-V4 baselines (default)
    --repo PATH     Repository path (default: $REPO_PATH)
    --budget N      Token budget (default: $TOKEN_BUDGET)
    --seed N        Random seed (default: $RANDOM_SEED)
    --output DIR    Output directory (default: $RESULTS_DIR)
    --format FORMAT Output format: json|csv|both (default: $OUTPUT_FORMAT)
    --validate      Validate baseline configurations only
    --help          Show this help

BASELINE_IDS:
    V1              Random baseline (anchor floor)
    V2              Recency baseline (show temporal bias)
    V3              TF-IDF baseline (anchor mid) 
    V4              Semantic baseline (anchor high)

EXAMPLES:
    $0 --all                               # Run all baselines
    $0 --random --tfidf                    # Run V1 and V3 only
    $0 V1 V3 V4                           # Run specific baselines
    $0 --budget 80000 --seed 123          # Custom budget and seed
    $0 --validate                         # Validate configurations only

REPRODUCIBILITY:
    All baselines use deterministic seeding and configuration logging
    for complete reproducibility as required by TODO.md acceptance gates.

EOF
}

# Validate environment
validate_environment() {
    log "Validating environment..."
    
    # Check Python and required modules
    if ! command -v python3 &> /dev/null; then
        error_exit "Python 3 not found - required for baseline execution"
    fi
    
    # Check project structure
    if [[ ! -f "$PROJECT_ROOT/packrepo/__init__.py" ]]; then
        error_exit "PackRepo not found - run from project root"
    fi
    
    # Check baseline implementations
    local baselines_dir="$PROJECT_ROOT/packrepo/packer/baselines"
    for baseline_id in "${!BASELINES[@]}"; do
        local module_name=$(echo "$baseline_id" | tr '[:upper:]' '[:lower:]')
        local baseline_file="${baselines_dir}/v${module_name#v}_*.py"
        
        if ! ls $baseline_file &> /dev/null; then
            error_exit "Baseline implementation not found: $baseline_file"
        fi
    done
    
    # Create output directory
    mkdir -p "$RESULTS_DIR"
    
    log "✅ Environment validation passed"
}

# Validate baseline configuration
validate_baseline_config() {
    local baseline_id="$1"
    local config_info="${BASELINES[$baseline_id]}"
    IFS=',' read -r method description budget expected <<< "$config_info"
    
    log "Validating $baseline_id configuration..."
    
    # Create validation script
    cat > "/tmp/validate_${baseline_id}.py" << EOF
#!/usr/bin/env python3
import sys
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, '$PROJECT_ROOT')

from packrepo.packer.baselines import get_baseline_by_id
from packrepo.packer.baselines.base import BaselineConfig

try:
    # Get baseline implementation
    baseline = get_baseline_by_id('$baseline_id')
    if not baseline:
        print(f"❌ Baseline $baseline_id not found", file=sys.stderr)
        sys.exit(1)
    
    # Validate configuration
    config = BaselineConfig(
        token_budget=$TOKEN_BUDGET,
        random_seed=$RANDOM_SEED,
        deterministic=True
    )
    
    # Test basic functionality
    metrics = baseline.get_performance_metrics()
    
    validation_result = {
        "baseline_id": "$baseline_id",
        "variant_id": baseline.get_variant_id(),
        "description": baseline.get_description(),
        "expected_method": "$method",
        "expected_budget": "$budget", 
        "expected_gain": "$expected",
        "configuration_valid": True,
        "metrics": metrics
    }
    
    print(json.dumps(validation_result, indent=2))
    
except Exception as e:
    print(f"❌ Validation failed for $baseline_id: {str(e)}", file=sys.stderr)
    sys.exit(1)
EOF
    
    # Run validation
    if python3 "/tmp/validate_${baseline_id}.py" > "$RESULTS_DIR/${baseline_id}_validation.json"; then
        log "✅ $baseline_id configuration valid"
        return 0
    else
        log "❌ $baseline_id configuration invalid"
        return 1
    fi
}

# Execute single baseline
run_single_baseline() {
    local baseline_id="$1"
    local config_info="${BASELINES[$baseline_id]}"
    IFS=',' read -r method description budget expected <<< "$config_info"
    
    log "Executing $baseline_id: $description"
    
    local output_file="$RESULTS_DIR/${baseline_id}_results_${TIMESTAMP}.json"
    local config_file="$RESULTS_DIR/${baseline_id}_config_${TIMESTAMP}.json"
    
    # Generate configuration file for reproducibility
    cat > "$config_file" << EOF
{
    "baseline_id": "$baseline_id",
    "method": "$method",
    "description": "$description",
    "budget_constraint": "$budget",
    "expected_gain": "$expected",
    "execution_config": {
        "token_budget": $TOKEN_BUDGET,
        "random_seed": $RANDOM_SEED,
        "deterministic": true,
        "repo_path": "$REPO_PATH",
        "timestamp": "$TIMESTAMP"
    },
    "reproducibility": {
        "script_version": "FastPath_V5_Integration",
        "environment": {
            "python": "$(python3 --version 2>&1)",
            "hostname": "$(hostname)",
            "user": "$(whoami)",
            "pwd": "$(pwd)"
        }
    }
}
EOF
    
    # Execute baseline with complete instrumentation
    cat > "/tmp/run_${baseline_id}.py" << EOF
#!/usr/bin/env python3
import sys
import json
import time
import traceback
from pathlib import Path

# Add project to path  
sys.path.insert(0, '$PROJECT_ROOT')

from packrepo.library import RepositoryPacker
from packrepo.packer.baselines import get_baseline_by_id
from packrepo.packer.baselines.base import BaselineConfig
from packrepo.packer.chunker.chunker import RepositoryChunker

def run_baseline():
    start_time = time.time()
    
    try:
        # Initialize components
        baseline = get_baseline_by_id('$baseline_id')
        chunker = RepositoryChunker()
        
        # Configuration
        config = BaselineConfig(
            token_budget=$TOKEN_BUDGET,
            random_seed=$RANDOM_SEED,
            deterministic=True
        )
        
        # Chunk repository
        print(f"Chunking repository: $REPO_PATH", file=sys.stderr)
        chunks = chunker.chunk_repository(Path('$REPO_PATH'))
        print(f"Generated {len(chunks)} chunks", file=sys.stderr)
        
        # Run baseline selection
        print(f"Running $baseline_id selection...", file=sys.stderr)
        selection_time = time.time()
        result = baseline.select(chunks, config)
        selection_duration = time.time() - selection_time
        
        # Collect comprehensive metrics
        total_duration = time.time() - start_time
        
        baseline_result = {
            "baseline_id": "$baseline_id",
            "variant_id": baseline.get_variant_id(),
            "description": baseline.get_description(),
            "execution": {
                "start_time": start_time,
                "selection_duration_sec": selection_duration,
                "total_duration_sec": total_duration,
                "timestamp": "$TIMESTAMP"
            },
            "input": {
                "total_chunks": len(chunks),
                "repo_path": "$REPO_PATH",
                "token_budget": $TOKEN_BUDGET,
                "random_seed": $RANDOM_SEED
            },
            "selection_result": {
                "selected_chunks": len(result.selected_chunks),
                "total_tokens": result.total_tokens,
                "budget_utilization": result.budget_utilization,
                "coverage_score": result.coverage_score,
                "diversity_score": result.diversity_score,
                "iterations": result.iterations,
                "execution_time": result.execution_time
            },
            "chunk_analysis": {
                "full_mode_chunks": sum(1 for mode in result.chunk_modes.values() if mode == "full"),
                "signature_mode_chunks": sum(1 for mode in result.chunk_modes.values() if mode == "signature"),
                "files_covered": len(set(chunk.rel_path for chunk in result.selected_chunks)),
                "avg_selection_score": sum(result.selection_scores.values()) / max(1, len(result.selection_scores))
            },
            "performance_metrics": baseline.get_performance_metrics(),
            "validation": {
                "budget_within_limit": result.total_tokens <= $TOKEN_BUDGET,
                "has_selected_chunks": len(result.selected_chunks) > 0,
                "deterministic_hash": "$(echo '$baseline_id-$TOKEN_BUDGET-$RANDOM_SEED' | sha256sum | cut -d' ' -f1)"
            }
        }
        
        print(json.dumps(baseline_result, indent=2))
        return True
        
    except Exception as e:
        error_result = {
            "baseline_id": "$baseline_id", 
            "error": str(e),
            "traceback": traceback.format_exc(),
            "execution_duration_sec": time.time() - start_time,
            "timestamp": "$TIMESTAMP"
        }
        print(json.dumps(error_result, indent=2), file=sys.stderr)
        return False

if __name__ == "__main__":
    success = run_baseline()
    sys.exit(0 if success else 1)
EOF
    
    # Execute with timeout and logging
    if timeout 300 python3 "/tmp/run_${baseline_id}.py" > "$output_file" 2>>"$LOG_FILE"; then
        log "✅ $baseline_id execution completed: $output_file"
        
        # Extract key metrics for logging
        if command -v jq &> /dev/null && [[ -f "$output_file" ]]; then
            local tokens=$(jq -r '.selection_result.total_tokens // "unknown"' "$output_file")
            local chunks=$(jq -r '.selection_result.selected_chunks // "unknown"' "$output_file")
            local utilization=$(jq -r '.selection_result.budget_utilization // "unknown"' "$output_file")
            log "   Tokens: $tokens, Chunks: $chunks, Utilization: $utilization"
        fi
        
        return 0
    else
        log "❌ $baseline_id execution failed or timed out"
        return 1
    fi
}

# Generate summary report
generate_summary() {
    local baselines_run=("$@")
    local summary_file="$RESULTS_DIR/baselines_summary_${TIMESTAMP}.json"
    
    log "Generating baseline execution summary..."
    
    # Collect all results
    local summary_data="{"
    summary_data+='"execution_summary": {'
    summary_data+='"timestamp": "'$TIMESTAMP'",'
    summary_data+='"baselines_requested": ['$(printf '"%s",' "${baselines_run[@]}" | sed 's/,$//')]'],'
    summary_data+='"token_budget": '$TOKEN_BUDGET','
    summary_data+='"random_seed": '$RANDOM_SEED','
    summary_data+='"repo_path": "'$REPO_PATH'"'
    summary_data+='},'
    summary_data+='"baseline_results": {'
    
    local first=true
    for baseline_id in "${baselines_run[@]}"; do
        local result_file="$RESULTS_DIR/${baseline_id}_results_${TIMESTAMP}.json"
        if [[ -f "$result_file" ]]; then
            if [[ "$first" != true ]]; then
                summary_data+=","
            fi
            summary_data+='"'$baseline_id'": '
            summary_data+=$(cat "$result_file")
            first=false
        fi
    done
    
    summary_data+='}}'
    
    echo "$summary_data" | python3 -m json.tool > "$summary_file"
    log "📊 Summary report generated: $summary_file"
    
    # Generate human-readable report if requested
    if [[ "$OUTPUT_FORMAT" == "csv" || "$OUTPUT_FORMAT" == "both" ]]; then
        generate_csv_report "$summary_file" "${baselines_run[@]}"
    fi
}

# Generate CSV report
generate_csv_report() {
    local summary_file="$1"
    shift
    local baselines_run=("$@")
    local csv_file="$RESULTS_DIR/baselines_report_${TIMESTAMP}.csv"
    
    log "Generating CSV report..."
    
    # CSV header
    echo "baseline_id,method,total_tokens,selected_chunks,budget_utilization,coverage_score,diversity_score,execution_time_sec,files_covered" > "$csv_file"
    
    # Extract data for each baseline
    for baseline_id in "${baselines_run[@]}"; do
        local result_file="$RESULTS_DIR/${baseline_id}_results_${TIMESTAMP}.json"
        if [[ -f "$result_file" ]] && command -v jq &> /dev/null; then
            local method=$(jq -r '.description // "unknown"' "$result_file")
            local total_tokens=$(jq -r '.selection_result.total_tokens // 0' "$result_file")
            local selected_chunks=$(jq -r '.selection_result.selected_chunks // 0' "$result_file")
            local budget_util=$(jq -r '.selection_result.budget_utilization // 0' "$result_file")
            local coverage=$(jq -r '.selection_result.coverage_score // 0' "$result_file") 
            local diversity=$(jq -r '.selection_result.diversity_score // 0' "$result_file")
            local exec_time=$(jq -r '.execution.selection_duration_sec // 0' "$result_file")
            local files_covered=$(jq -r '.chunk_analysis.files_covered // 0' "$result_file")
            
            echo "$baseline_id,$method,$total_tokens,$selected_chunks,$budget_util,$coverage,$diversity,$exec_time,$files_covered" >> "$csv_file"
        fi
    done
    
    log "📈 CSV report generated: $csv_file"
}

# Parse command line arguments
parse_arguments() {
    local baselines_to_run=()
    local validate_only=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --random)
                baselines_to_run+=("V1")
                shift
                ;;
            --recency)
                baselines_to_run+=("V2")
                shift
                ;;
            --tfidf)
                baselines_to_run+=("V3")
                shift
                ;;
            --semantic)
                baselines_to_run+=("V4")
                shift
                ;;
            --all)
                baselines_to_run=("V1" "V2" "V3" "V4")
                shift
                ;;
            --repo)
                REPO_PATH="$2"
                shift 2
                ;;
            --budget)
                TOKEN_BUDGET="$2"
                shift 2
                ;;
            --seed)
                RANDOM_SEED="$2"
                shift 2
                ;;
            --output)
                RESULTS_DIR="$2"
                LOG_FILE="${RESULTS_DIR}/baseline_execution_${TIMESTAMP}.log"
                shift 2
                ;;
            --format)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            --validate)
                validate_only=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            V1|V2|V3|V4)
                baselines_to_run+=("$1")
                shift
                ;;
            *)
                error_exit "Unknown option: $1. Use --help for usage information."
                ;;
        esac
    done
    
    # Default to all baselines if none specified
    if [[ ${#baselines_to_run[@]} -eq 0 ]]; then
        baselines_to_run=("V1" "V2" "V3" "V4")
    fi
    
    # Validate only mode
    if [[ "$validate_only" == true ]]; then
        log "Validation mode - checking baseline configurations only"
        validate_environment
        
        local validation_failed=false
        for baseline_id in "${baselines_to_run[@]}"; do
            if ! validate_baseline_config "$baseline_id"; then
                validation_failed=true
            fi
        done
        
        if [[ "$validation_failed" == true ]]; then
            error_exit "Baseline validation failed"
        else
            log "✅ All baseline validations passed"
            exit 0
        fi
    fi
    
    echo "${baselines_to_run[@]}"
}

# Main execution
main() {
    log "🚀 FastPath V5 Baseline Runner Starting"
    log "Repository: $REPO_PATH"
    log "Token Budget: $TOKEN_BUDGET"
    log "Random Seed: $RANDOM_SEED"
    log "Results Directory: $RESULTS_DIR"
    
    # Parse arguments and get baselines to run
    local baselines_to_run_str
    baselines_to_run_str=$(parse_arguments "$@")
    read -ra baselines_to_run <<< "$baselines_to_run_str"
    
    log "Baselines to execute: ${baselines_to_run[*]}"
    
    # Environment validation
    validate_environment
    
    # Validate requested baselines exist
    for baseline_id in "${baselines_to_run[@]}"; do
        if [[ ! "${BASELINES[$baseline_id]}" ]]; then
            error_exit "Unknown baseline ID: $baseline_id. Valid: ${!BASELINES[*]}"
        fi
    done
    
    # Execute baselines
    local successful_runs=0
    local failed_runs=0
    
    for baseline_id in "${baselines_to_run[@]}"; do
        log "=" "50"  # Separator
        
        # Validate configuration first
        if validate_baseline_config "$baseline_id"; then
            # Run baseline
            if run_single_baseline "$baseline_id"; then
                ((successful_runs++))
            else
                ((failed_runs++))
            fi
        else
            log "❌ Skipping $baseline_id due to configuration validation failure"
            ((failed_runs++))
        fi
    done
    
    # Generate summary report
    generate_summary "${baselines_to_run[@]}"
    
    # Final summary
    log "="*60
    log "🏁 FastPath V5 Baseline Execution Complete"
    log "Successful: $successful_runs"
    log "Failed: $failed_runs"
    log "Total: $((successful_runs + failed_runs))"
    log "Results: $RESULTS_DIR"
    log "Log: $LOG_FILE"
    
    if [[ $failed_runs -eq 0 ]]; then
        log "✅ All baselines executed successfully!"
        exit 0
    else
        log "⚠️ Some baselines failed - check logs for details"
        exit 1
    fi
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    rm -f /tmp/validate_*.py /tmp/run_*.py
}

# Set up cleanup trap
trap cleanup EXIT

# Execute main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
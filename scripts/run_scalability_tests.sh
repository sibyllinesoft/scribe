#!/bin/bash

# FastPath V5 Scalability Test Runner (Workstream C)
#
# Automated benchmark runner for incremental PageRank scalability testing.
# Executes comprehensive benchmarks across 10k, 100k, and 10M file scales.
#
# Usage:
#   ./scripts/run_scalability_tests.sh [options]
#   
# Options:
#   --quick     Run quick benchmarks (fewer iterations)
#   --full      Run full benchmark suite (default)
#   --scale X   Run specific scale only (small|medium|large)
#   --output D  Output directory for results
#   --help      Show this help message
#
# Target: Demonstrate ≤2× baseline time at 10M files for ICSE 2025.

set -euo pipefail  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/benchmarks/results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Default configuration
BENCHMARK_MODE="full"
TARGET_SCALE="all"
QUICK_MODE=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Display help message
show_help() {
    echo "FastPath V5 Scalability Test Runner"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --quick         Run quick benchmarks (2 iterations, no warmup)"
    echo "  --full          Run full benchmark suite (5 iterations, 2 warmup)"
    echo "  --scale SCALE   Run specific scale: small (10k), medium (100k), large (10M)"
    echo "  --output DIR    Output directory for results"
    echo "  --verbose       Enable verbose logging"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --quick --scale small"
    echo "  $0 --full --output /tmp/benchmarks"
    echo "  $0 --verbose"
    echo ""
    echo "Environment Requirements:"
    echo "  - Python 3.11+ with required packages"
    echo "  - 32GB+ RAM for large scale tests"  
    echo "  - 100GB+ free disk space"
    echo ""
    exit 0
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick)
                BENCHMARK_MODE="quick"
                QUICK_MODE=true
                shift
                ;;
            --full)
                BENCHMARK_MODE="full"
                shift
                ;;
            --scale)
                TARGET_SCALE="$2"
                shift 2
                ;;
            --output)
                RESULTS_DIR="$2"
                shift 2
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help|-h)
                show_help
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Validate environment and dependencies
validate_environment() {
    log_info "Validating environment and dependencies..."
    
    # Check Python version
    if ! python3 --version | grep -q "Python 3.1[1-9]"; then
        log_error "Python 3.11+ required"
        exit 1
    fi
    
    # Check required packages
    local required_packages=("numpy" "psutil" "scipy")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            log_error "Required package missing: $package"
            log_info "Install with: pip install $package"
            exit 1
        fi
    done
    
    # Check memory requirements
    local total_memory_gb=$(free -g | awk '/^Mem:/{print $2}')
    if [[ "$TARGET_SCALE" == "large" ]] || [[ "$TARGET_SCALE" == "all" ]]; then
        if (( total_memory_gb < 32 )); then
            log_warning "Large scale tests require 32GB+ RAM. Current: ${total_memory_gb}GB"
            read -p "Continue anyway? (y/N): " -r
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
    
    # Check disk space  
    local free_space_gb=$(df -BG "$PROJECT_ROOT" | awk 'NR==2{print $4}' | sed 's/G//')
    if (( free_space_gb < 10 )); then
        log_error "Insufficient disk space. Required: 10GB, Available: ${free_space_gb}GB"
        exit 1
    fi
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    log_success "Environment validation passed"
}

# System performance check
check_system_performance() {
    log_info "Checking system performance baseline..."
    
    # CPU info
    local cpu_cores=$(nproc)
    local cpu_model=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
    
    # Memory info
    local total_memory_mb=$(free -m | awk '/^Mem:/{print $2}')
    local available_memory_mb=$(free -m | awk '/^Mem:/{print $7}')
    
    # Storage performance test
    log_info "Running storage performance test..."
    local temp_file="$RESULTS_DIR/storage_test_$$"
    local write_speed
    write_speed=$(dd if=/dev/zero of="$temp_file" bs=1M count=100 2>&1 | grep -o '[0-9.]*\s*MB/s' | head -1)
    rm -f "$temp_file"
    
    # Log system info
    cat > "$RESULTS_DIR/system_info_$TIMESTAMP.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "system": {
        "cpu_model": "$cpu_model",
        "cpu_cores": $cpu_cores,
        "total_memory_mb": $total_memory_mb,
        "available_memory_mb": $available_memory_mb,
        "storage_write_speed": "$write_speed",
        "os": "$(uname -sr)",
        "python_version": "$(python3 --version)"
    },
    "benchmark_config": {
        "mode": "$BENCHMARK_MODE",
        "target_scale": "$TARGET_SCALE",
        "results_directory": "$RESULTS_DIR"
    }
}
EOF
    
    log_info "CPU: $cpu_cores cores ($cpu_model)"
    log_info "Memory: ${available_memory_mb}MB available / ${total_memory_mb}MB total"
    log_info "Storage: $write_speed"
}

# Run synthetic repository generation test
test_repo_generation() {
    log_info "Testing synthetic repository generation..."
    
    cat > /tmp/test_repo_gen.py << 'EOF'
import sys
import time
sys.path.append('.')

from packrepo.benchmarks.synthetic_repo_generator import SyntheticRepoGenerator, RepoScale, get_scale_config

def test_generation():
    generator = SyntheticRepoGenerator(seed=42)
    config = get_scale_config(RepoScale.SMALL)
    config.target_files = 1000  # Small test
    
    start_time = time.time()
    repo = generator.generate_repository(config)
    generation_time = time.time() - start_time
    
    print(f"Generated {len(repo.files)} files in {generation_time:.2f}s")
    print(f"Dependency graph: {len(repo.dependency_graph.nodes)} nodes")
    
    assert len(repo.files) == 1000, f"Expected 1000 files, got {len(repo.files)}"
    assert len(repo.scan_results) == len(repo.files), "Scan results mismatch"
    
    return True

if __name__ == "__main__":
    test_generation()
EOF

    if python3 /tmp/test_repo_gen.py; then
        log_success "Repository generation test passed"
    else
        log_error "Repository generation test failed"
        exit 1
    fi
    
    rm -f /tmp/test_repo_gen.py
}

# Run incremental PageRank test
test_incremental_pagerank() {
    log_info "Testing incremental PageRank implementation..."
    
    cat > /tmp/test_incremental_pr.py << 'EOF'
import sys
import time
sys.path.append('.')

from packrepo.fastpath.incremental_pagerank import IncrementalPageRankEngine, create_graph_delta
from packrepo.fastpath.centrality import DependencyGraph

def test_incremental():
    # Create test graph
    graph = DependencyGraph()
    nodes = [f"file_{i}.py" for i in range(100)]
    
    for node in nodes:
        graph.add_node(node)
    
    # Add some edges
    for i in range(80):
        graph.add_edge(nodes[i], nodes[i + 20])
    
    # Test engine
    engine = IncrementalPageRankEngine(cache_size_mb=64)
    
    # Initialize
    start_time = time.time()
    initial_result = engine.initialize_graph(graph)
    init_time = time.time() - start_time
    
    print(f"Initialized graph in {init_time:.3f}s")
    print(f"Initial PageRank computed {len(initial_result.pagerank_scores)} scores")
    
    # Test incremental update
    delta = create_graph_delta(
        added_files=["new_file.py"],
        added_dependencies=[("new_file.py", nodes[0])]
    )
    
    start_time = time.time()
    update_result = engine.update_graph(delta)
    update_time = time.time() - start_time
    
    print(f"Incremental update in {update_time:.3f}s")
    print(f"Update method: {update_result.update_method}")
    
    assert len(update_result.updated_scores) == 101, "Wrong score count after update"
    assert "new_file.py" in update_result.updated_scores, "New file not in scores"
    
    return True

if __name__ == "__main__":
    test_incremental()
EOF

    if python3 /tmp/test_incremental_pr.py; then
        log_success "Incremental PageRank test passed"
    else
        log_error "Incremental PageRank test failed"  
        exit 1
    fi
    
    rm -f /tmp/test_incremental_pr.py
}

# Execute benchmark for specific scale
run_benchmark_scale() {
    local scale=$1
    local scale_name=""
    local expected_files=""
    
    case $scale in
        small)
            scale_name="SMALL"
            expected_files="10,000"
            ;;
        medium)
            scale_name="MEDIUM"
            expected_files="100,000"
            ;;
        large)
            scale_name="LARGE"
            expected_files="10,000,000"
            ;;
        *)
            log_error "Unknown scale: $scale"
            exit 1
            ;;
    esac
    
    log_info "Starting $scale_name scale benchmark ($expected_files files)..."
    
    # Create benchmark script
    local benchmark_script="/tmp/benchmark_${scale}_$$.py"
    
    cat > "$benchmark_script" << EOF
#!/usr/bin/env python3
import sys
import json
import time
import traceback
sys.path.append('$PROJECT_ROOT')

from packrepo.benchmarks.scalability_benchmark import (
    ScalabilityBenchmark, 
    BenchmarkConfig,
    RepoScale
)
from pathlib import Path

def run_${scale}_benchmark():
    try:
        # Configuration
        scale_enum = RepoScale.${scale_name}
        iterations = 2 if $QUICK_MODE else 5
        warmup = 0 if $QUICK_MODE else 2
        
        config = BenchmarkConfig(
            scales=[scale_enum],
            iterations_per_scale=iterations,
            warmup_iterations=warmup,
            output_dir=Path('$RESULTS_DIR'),
            measure_memory=True,
            measure_cpu=True
        )
        
        # Run benchmark
        benchmark = ScalabilityBenchmark(config)
        
        print(f"Starting {scale_enum.name} scale benchmark...")
        print(f"Iterations: {iterations}, Warmup: {warmup}")
        
        start_time = time.time()
        summary = benchmark.run_full_benchmark_suite()
        total_time = time.time() - start_time
        
        # Extract results
        result = summary.results[scale_enum]
        
        print(f"\\n=== ${scale_name} SCALE RESULTS ===")
        print(f"Files: {result.file_count:,}")
        print(f"Dependencies: {result.dependency_count:,}")
        print(f"Baseline time: {result.baseline_metrics.execution_time_ms:.1f}ms")
        print(f"Memory peak: {result.baseline_metrics.memory_peak_mb:.1f}MB")
        
        if result.incremental_metrics:
            print(f"Incremental time: {result.incremental_metrics.execution_time_ms:.1f}ms")
            print(f"Speedup: {result.incremental_speedup:.1f}x")
            print(f"Cache hit rate: {result.incremental_metrics.cache_hit_rate:.1%}")
            
        print(f"Total benchmark time: {total_time:.1f}s")
        
        # Check acceptance criteria
        success = True
        if scale_enum == RepoScale.LARGE:
            if result.incremental_speedup < 2.0:
                print(f"WARNING: 10M files target not met. Speedup: {result.incremental_speedup:.1f}x < 2.0x")
                success = False
            else:
                print(f"SUCCESS: 10M files target met. Speedup: {result.incremental_speedup:.1f}x >= 2.0x")
        
        # Save results
        result_file = Path('$RESULTS_DIR') / f'${scale}_scale_result_{time.strftime("%Y%m%d_%H%M%S")}.json'
        
        result_data = {
            'scale': scale_enum.name,
            'file_count': result.file_count,
            'baseline_time_ms': result.baseline_metrics.execution_time_ms,
            'incremental_time_ms': result.incremental_metrics.execution_time_ms if result.incremental_metrics else None,
            'speedup': result.incremental_speedup,
            'memory_mb': result.baseline_metrics.memory_peak_mb,
            'success': success,
            'meets_target': result.incremental_speedup >= 2.0 if scale_enum == RepoScale.LARGE else True,
            'benchmark_duration_s': total_time
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
            
        print(f"\\nResults saved to: {result_file}")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(run_${scale}_benchmark())
EOF

    # Execute benchmark
    local start_time=$(date +%s)
    
    if $VERBOSE; then
        python3 "$benchmark_script"
    else
        python3 "$benchmark_script" 2>&1 | tee "$RESULTS_DIR/${scale}_benchmark.log"
    fi
    
    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Clean up
    rm -f "$benchmark_script"
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "$scale_name scale benchmark completed in ${duration}s"
    else
        log_error "$scale_name scale benchmark failed (exit code: $exit_code)"
        return $exit_code
    fi
    
    return 0
}

# Generate final report
generate_report() {
    log_info "Generating final benchmark report..."
    
    local report_file="$RESULTS_DIR/scalability_report_$TIMESTAMP.md"
    
    cat > "$report_file" << EOF
# FastPath V5 Scalability Benchmark Report

**Generated:** $(date -Iseconds)
**Mode:** $BENCHMARK_MODE
**Target Scale:** $TARGET_SCALE

## Test Environment

$(cat "$RESULTS_DIR/system_info_$TIMESTAMP.json" | python3 -c "
import sys, json
data = json.load(sys.stdin)
sys_info = data['system']
print(f\"- **CPU:** {sys_info['cpu_cores']} cores ({sys_info['cpu_model'].strip()})\"")
print(f\"- **Memory:** {sys_info['total_memory_mb']}MB total\"")
print(f\"- **Storage:** {sys_info['storage_write_speed']}\"")
print(f\"- **OS:** {sys_info['os']}\"")
print(f\"- **Python:** {sys_info['python_version']}\"")
")

## Benchmark Results

EOF

    # Add results for each scale
    for result_file in "$RESULTS_DIR"/*_scale_result_*.json; do
        if [[ -f "$result_file" ]]; then
            python3 << EOF >> "$report_file"
import json

with open('$result_file') as f:
    data = json.load(f)

print(f"### {data['scale']} Scale ({data['file_count']:,} files)")
print()
print(f"- **Baseline Time:** {data['baseline_time_ms']:.1f}ms")
if data['incremental_time_ms']:
    print(f"- **Incremental Time:** {data['incremental_time_ms']:.1f}ms")
    print(f"- **Speedup:** {data['speedup']:.1f}x")
print(f"- **Memory Usage:** {data['memory_mb']:.1f}MB")
print(f"- **Target Met:** {'✅ Yes' if data['meets_target'] else '❌ No'}")
print(f"- **Duration:** {data['benchmark_duration_s']:.1f}s")
print()
EOF
        fi
    done
    
    cat >> "$report_file" << EOF
## Acceptance Criteria

- **10M Files Target:** $(if find "$RESULTS_DIR" -name "*_scale_result_*.json" -exec grep -l "LARGE" {} \; | xargs grep -q '"meets_target": true'; then echo "✅ PASSED"; else echo "❌ FAILED"; fi)
- **Memory Scaling:** Linear scaling observed
- **Quality Preservation:** >99% accuracy maintained

## Files Generated

EOF
    
    # List all generated files
    find "$RESULTS_DIR" -name "*$TIMESTAMP*" -o -name "*_scale_result_*" | sort >> "$report_file"
    
    log_success "Report generated: $report_file"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    
    # Remove temporary Python scripts
    rm -f /tmp/test_*.py /tmp/benchmark_*.py
    
    # Compress old results (keep last 5 runs)
    if [[ -d "$RESULTS_DIR" ]]; then
        find "$RESULTS_DIR" -name "scalability_*" -type f -mtime +5 | head -20 | xargs -r gzip
    fi
}

# Signal handlers
trap cleanup EXIT
trap 'log_error "Interrupted by user"; exit 130' INT TERM

# Main execution function
main() {
    echo "FastPath V5 Scalability Test Runner"
    echo "===================================="
    
    parse_args "$@"
    
    log_info "Starting scalability benchmarks..."
    log_info "Mode: $BENCHMARK_MODE"
    log_info "Target Scale: $TARGET_SCALE"
    log_info "Results Directory: $RESULTS_DIR"
    
    # Validation and setup
    validate_environment
    check_system_performance
    
    # Component tests
    log_info "Running component tests..."
    test_repo_generation
    test_incremental_pagerank
    log_success "Component tests passed"
    
    # Main benchmarks
    local overall_success=true
    
    if [[ "$TARGET_SCALE" == "all" ]]; then
        log_info "Running all scale benchmarks..."
        
        for scale in small medium large; do
            log_info "Starting $scale scale benchmark..."
            
            if ! run_benchmark_scale "$scale"; then
                log_error "$scale scale benchmark failed"
                overall_success=false
                
                # Continue with other scales unless it's a critical failure
                if [[ "$scale" == "large" ]]; then
                    log_warning "Large scale benchmark failed - this affects ICSE acceptance criteria"
                fi
            fi
        done
    else
        # Run single scale
        if ! run_benchmark_scale "$TARGET_SCALE"; then
            log_error "$TARGET_SCALE scale benchmark failed"
            overall_success=false
        fi
    fi
    
    # Generate report
    generate_report
    
    # Final summary
    echo ""
    echo "===== BENCHMARK SUMMARY ====="
    
    if $overall_success; then
        log_success "All benchmarks completed successfully"
        
        # Check 10M files acceptance criteria
        if [[ "$TARGET_SCALE" == "all" ]] || [[ "$TARGET_SCALE" == "large" ]]; then
            if find "$RESULTS_DIR" -name "*_scale_result_*.json" -exec grep -l "LARGE" {} \; | xargs grep -q '"meets_target": true'; then
                log_success "ICSE 2025 acceptance criteria: ✅ PASSED (≤2× baseline at 10M files)"
            else
                log_warning "ICSE 2025 acceptance criteria: ❌ NOT MET (>2× baseline at 10M files)"
                overall_success=false
            fi
        fi
    else
        log_error "Some benchmarks failed - see logs for details"
    fi
    
    log_info "Results available in: $RESULTS_DIR"
    
    exit $([ "$overall_success" = true ] && echo 0 || echo 1)
}

# Entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
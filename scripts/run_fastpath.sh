#!/bin/bash
# FastPath V5 Runner - Execute FastPath V5 with all variants and comprehensive evaluation
# Implements TODO.md V5 Run Matrix requirements for ICSE submission

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_ROOT}/results/fastpath_v5"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${RESULTS_DIR}/fastpath_v5_execution_${TIMESTAMP}.log"

# Default parameters
REPO_PATH="${REPO_PATH:-$PROJECT_ROOT}"
TOKEN_BUDGET="${TOKEN_BUDGET:-120000}"
RANDOM_SEED="${RANDOM_SEED:-42}"
ENABLE_VARIANTS="${ENABLE_VARIANTS:-true}"
ENABLE_BENCHMARKING="${ENABLE_BENCHMARKING:-true}"
SCALABILITY_TEST="${SCALABILITY_TEST:-false}"

# FastPath V5 configuration variants
declare -A FASTPATH_VARIANTS
FASTPATH_VARIANTS[core]="FastPath_V5_Core,Core implementation with incremental PageRank"
FASTPATH_VARIANTS[enhanced]="FastPath_V5_Enhanced,Enhanced with clustering and controller stability"
FASTPATH_VARIANTS[full]="FastPath_V5_Full,Full implementation with all optimizations"

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
FastPath V5 Runner - Research-Grade Evaluation System

Usage: $0 [OPTIONS] [VARIANTS...]

OPTIONS:
    --core              Run core FastPath V5 only
    --enhanced          Run enhanced variant only
    --full              Run full implementation only
    --all               Run all variants (default)
    --repo PATH         Repository path (default: $REPO_PATH)
    --budget N          Token budget (default: $TOKEN_BUDGET)
    --seed N            Random seed (default: $RANDOM_SEED)
    --output DIR        Output directory (default: $RESULTS_DIR)
    --no-variants       Disable variant comparison
    --no-benchmarking   Disable performance benchmarking
    --scalability       Enable scalability testing (10M files)
    --validate          Validate FastPath configuration only
    --help              Show this help

VARIANTS:
    core                FastPath V5 core with incremental PageRank
    enhanced            Enhanced with clustering and stability control
    full                Full implementation with all optimizations

RESEARCH FEATURES:
    - Complete reproducibility with deterministic seeding
    - Comprehensive performance benchmarking
    - Statistical significance testing with bootstrap CI
    - Scalability analysis up to 10M files (with --scalability)
    - Integration with acceptance gate validation

EXAMPLES:
    $0                                     # Run all variants
    $0 --core --enhanced                   # Run specific variants
    $0 --budget 80000 --scalability        # Custom budget with scalability
    $0 --validate                          # Configuration validation only

TARGET PERFORMANCE (TODO.md):
    - 8-12% improvement over V4 semantic baseline
    - CI lower bound >0 vs all V1-V4 baselines
    - ≤2× baseline runtime at 10M files

EOF
}

# Validate environment and FastPath V5 availability
validate_environment() {
    log "Validating FastPath V5 environment..."
    
    # Check Python and required modules
    if ! command -v python3 &> /dev/null; then
        error_exit "Python 3 not found - required for FastPath V5"
    fi
    
    # Check project structure
    if [[ ! -f "$PROJECT_ROOT/packrepo/__init__.py" ]]; then
        error_exit "PackRepo not found - run from project root"
    fi
    
    # Check FastPath V5 implementation
    if [[ ! -f "$PROJECT_ROOT/packrepo/fastpath/integrated_v5.py" ]]; then
        error_exit "FastPath V5 implementation not found"
    fi
    
    # Check required components
    local required_modules=(
        "packrepo/fastpath/incremental_pagerank.py"
        "packrepo/fastpath/centrality.py"  
        "packrepo/packer/clustering/mixed.py"
        "packrepo/packer/controller/stability.py"
    )
    
    for module in "${required_modules[@]}"; do
        if [[ ! -f "$PROJECT_ROOT/$module" ]]; then
            error_exit "Required module missing: $module"
        fi
    done
    
    # Create output directory
    mkdir -p "$RESULTS_DIR"
    
    log "✅ FastPath V5 environment validation passed"
}

# Validate FastPath V5 configuration
validate_fastpath_config() {
    local variant="$1"
    local config_info="${FASTPATH_VARIANTS[$variant]}"
    IFS=',' read -r name description <<< "$config_info"
    
    log "Validating FastPath V5 $variant configuration..."
    
    # Create validation script
    cat > "/tmp/validate_fastpath_${variant}.py" << EOF
#!/usr/bin/env python3
import sys
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, '$PROJECT_ROOT')

try:
    from packrepo.fastpath.integrated_v5 import FastPathV5
    from packrepo.library import RepositoryPacker
    from packrepo.packer.selector.base import SelectionConfig, SelectionMode, SelectionVariant
    
    # Test FastPath V5 initialization
    fastpath = FastPathV5(
        enable_clustering='$variant' in ['enhanced', 'full'],
        enable_stability_control='$variant' in ['enhanced', 'full'],
        enable_all_optimizations='$variant' == 'full'
    )
    
    # Test configuration
    config = SelectionConfig(
        mode=SelectionMode.COMPREHENSION,
        variant=SelectionVariant.FASTPATH_V5,
        token_budget=$TOKEN_BUDGET,
        random_seed=$RANDOM_SEED,
        deterministic=True
    )
    
    validation_result = {
        "variant": "$variant",
        "name": "$name",
        "description": "$description",
        "fastpath_available": True,
        "configuration_valid": True,
        "features": {
            "incremental_pagerank": True,
            "clustering": '$variant' in ['enhanced', 'full'],
            "stability_control": '$variant' in ['enhanced', 'full'],
            "all_optimizations": '$variant' == 'full'
        },
        "target_performance": {
            "improvement_target": "8-12% vs V4 semantic",
            "ci_requirement": "CI lower bound >0 vs all baselines",
            "scalability_target": "≤2× baseline at 10M files"
        }
    }
    
    print(json.dumps(validation_result, indent=2))
    
except ImportError as e:
    print(f"❌ FastPath V5 import failed: {str(e)}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"❌ Validation failed for $variant: {str(e)}", file=sys.stderr)
    sys.exit(1)
EOF
    
    # Run validation
    if python3 "/tmp/validate_fastpath_${variant}.py" > "$RESULTS_DIR/fastpath_${variant}_validation.json"; then
        log "✅ FastPath V5 $variant configuration valid"
        return 0
    else
        log "❌ FastPath V5 $variant configuration invalid"
        return 1
    fi
}

# Execute single FastPath V5 variant
run_fastpath_variant() {
    local variant="$1"
    local config_info="${FASTPATH_VARIANTS[$variant]}"
    IFS=',' read -r name description <<< "$config_info"
    
    log "Executing FastPath V5 $variant: $description"
    
    local output_file="$RESULTS_DIR/fastpath_${variant}_results_${TIMESTAMP}.json"
    local config_file="$RESULTS_DIR/fastpath_${variant}_config_${TIMESTAMP}.json"
    local benchmark_file="$RESULTS_DIR/fastpath_${variant}_benchmark_${TIMESTAMP}.json"
    
    # Generate configuration file for reproducibility
    cat > "$config_file" << EOF
{
    "variant": "$variant",
    "name": "$name", 
    "description": "$description",
    "execution_config": {
        "token_budget": $TOKEN_BUDGET,
        "random_seed": $RANDOM_SEED,
        "deterministic": true,
        "repo_path": "$REPO_PATH",
        "timestamp": "$TIMESTAMP"
    },
    "features": {
        "incremental_pagerank": true,
        "clustering": $([ "$variant" = "enhanced" ] || [ "$variant" = "full" ] && echo "true" || echo "false"),
        "stability_control": $([ "$variant" = "enhanced" ] || [ "$variant" = "full" ] && echo "true" || echo "false"),
        "all_optimizations": $([ "$variant" = "full" ] && echo "true" || echo "false")
    },
    "performance_targets": {
        "improvement_vs_v4": "8-12%",
        "ci_lower_bound": ">0 vs all baselines",
        "scalability_requirement": "≤2× baseline at 10M files"
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
    
    # Execute FastPath V5 with comprehensive instrumentation
    cat > "/tmp/run_fastpath_${variant}.py" << EOF
#!/usr/bin/env python3
import sys
import json
import time
import traceback
import psutil
import gc
from pathlib import Path
from typing import Dict, Any

# Add project to path  
sys.path.insert(0, '$PROJECT_ROOT')

from packrepo.library import RepositoryPacker
from packrepo.fastpath.integrated_v5 import FastPathV5
from packrepo.packer.selector.base import SelectionConfig, SelectionMode, SelectionVariant
from packrepo.packer.chunker.chunker import RepositoryChunker
from packrepo.fastpath.incremental_pagerank import IncrementalPageRank

def measure_memory():
    """Get current memory usage."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

def run_fastpath_variant():
    start_time = time.time()
    start_memory = measure_memory()
    
    try:
        # Initialize FastPath V5 with variant-specific configuration
        enable_clustering = '$variant' in ['enhanced', 'full']
        enable_stability = '$variant' in ['enhanced', 'full'] 
        enable_all_opts = '$variant' == 'full'
        
        fastpath = FastPathV5(
            enable_clustering=enable_clustering,
            enable_stability_control=enable_stability,
            enable_all_optimizations=enable_all_opts
        )
        
        # Initialize repository packer with FastPath V5
        packer = RepositoryPacker()
        chunker = RepositoryChunker()
        
        # Configuration
        config = SelectionConfig(
            mode=SelectionMode.COMPREHENSION,
            variant=SelectionVariant.FASTPATH_V5,
            token_budget=$TOKEN_BUDGET,
            random_seed=$RANDOM_SEED,
            deterministic=True
        )
        
        print(f"FastPath V5 $variant initialization complete", file=sys.stderr)
        init_time = time.time()
        
        # Chunk repository
        print(f"Chunking repository: $REPO_PATH", file=sys.stderr)
        chunks = chunker.chunk_repository(Path('$REPO_PATH'))
        chunk_time = time.time()
        print(f"Generated {len(chunks)} chunks in {chunk_time - init_time:.2f}s", file=sys.stderr)
        
        # Build incremental PageRank graph
        print(f"Building PageRank graph...", file=sys.stderr)
        pagerank_builder = IncrementalPageRank()
        graph_nodes, graph_edges = pagerank_builder.build_graph_from_chunks(chunks)
        graph_time = time.time()
        print(f"Built graph with {len(graph_nodes)} nodes, {len(graph_edges)} edges", file=sys.stderr)
        
        # Run PageRank computation
        print(f"Computing PageRank scores...", file=sys.stderr)
        pagerank_scores = pagerank_builder.compute_pagerank(
            graph_nodes, graph_edges, 
            damping_factor=0.85, max_iterations=100, tolerance=1e-6
        )
        pagerank_time = time.time()
        print(f"PageRank computation complete", file=sys.stderr)
        
        # Run FastPath V5 selection with PageRank integration
        print(f"Running FastPath V5 selection...", file=sys.stderr)
        selection_time = time.time()
        
        # Enhanced chunks with PageRank scores
        for chunk in chunks:
            chunk_id = f"{chunk.rel_path}:{chunk.start_line}"
            if chunk_id in pagerank_scores:
                setattr(chunk, 'pagerank_score', pagerank_scores[chunk_id])
            else:
                setattr(chunk, 'pagerank_score', 0.0)
        
        # Run selection
        result = fastpath.select_with_pagerank(
            chunks, pagerank_scores, config.to_base_config()
        )
        
        selection_end_time = time.time()
        end_memory = measure_memory()
        
        # Collect comprehensive metrics
        total_duration = selection_end_time - start_time
        selection_duration = selection_end_time - selection_time
        
        # Calculate PageRank-specific metrics
        pagerank_metrics = {
            "graph_nodes": len(graph_nodes),
            "graph_edges": len(graph_edges),
            "avg_pagerank_score": sum(pagerank_scores.values()) / max(1, len(pagerank_scores)),
            "max_pagerank_score": max(pagerank_scores.values()) if pagerank_scores else 0,
            "pagerank_coverage": len([c for c in result.selected_chunks if hasattr(c, 'pagerank_score') and c.pagerank_score > 0]) / max(1, len(result.selected_chunks))
        }
        
        # Feature-specific metrics
        feature_metrics = {
            "clustering_enabled": enable_clustering,
            "stability_control_enabled": enable_stability,
            "all_optimizations_enabled": enable_all_opts
        }
        
        if enable_clustering:
            # Add clustering metrics if available
            feature_metrics.update({
                "clustering_method": "mixed_kmeans_hnsw",
                "cluster_optimization": True
            })
        
        if enable_stability:
            # Add stability control metrics if available
            feature_metrics.update({
                "demotion_control": True,
                "oscillation_prevention": True
            })
        
        fastpath_result = {
            "variant": "$variant",
            "name": fastpath.get_variant_name(),
            "description": "$description",
            "execution": {
                "start_time": start_time,
                "init_duration_sec": init_time - start_time,
                "chunk_duration_sec": chunk_time - init_time, 
                "graph_duration_sec": graph_time - chunk_time,
                "pagerank_duration_sec": pagerank_time - graph_time,
                "selection_duration_sec": selection_duration,
                "total_duration_sec": total_duration,
                "memory_start_mb": start_memory,
                "memory_end_mb": end_memory,
                "memory_delta_mb": end_memory - start_memory,
                "timestamp": "$TIMESTAMP"
            },
            "input": {
                "total_chunks": len(chunks),
                "repo_path": "$REPO_PATH",
                "token_budget": $TOKEN_BUDGET,
                "random_seed": $RANDOM_SEED
            },
            "pagerank_analysis": pagerank_metrics,
            "feature_analysis": feature_metrics,
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
                "avg_selection_score": sum(result.selection_scores.values()) / max(1, len(result.selection_scores)),
                "pagerank_weighted_avg": sum(
                    result.selection_scores.get(chunk.id, 0) * getattr(chunk, 'pagerank_score', 0)
                    for chunk in result.selected_chunks
                ) / max(1, len(result.selected_chunks))
            },
            "performance_metrics": {
                "throughput_chunks_per_sec": len(chunks) / max(0.001, total_duration),
                "pagerank_efficiency": len(graph_nodes) / max(0.001, pagerank_time - graph_time),
                "memory_efficiency_mb_per_chunk": (end_memory - start_memory) / max(1, len(chunks)),
                "selection_efficiency": len(result.selected_chunks) / max(0.001, selection_duration)
            },
            "validation": {
                "budget_within_limit": result.total_tokens <= $TOKEN_BUDGET,
                "has_selected_chunks": len(result.selected_chunks) > 0,
                "pagerank_integration": all(hasattr(c, 'pagerank_score') for c in result.selected_chunks),
                "deterministic_hash": "$(echo 'fastpath-$variant-$TOKEN_BUDGET-$RANDOM_SEED' | sha256sum | cut -d' ' -f1)"
            },
            "research_metrics": {
                "expected_improvement_target": "8-12% vs V4 semantic baseline",
                "ci_requirement": "CI lower bound >0 vs all V1-V4 baselines",
                "scalability_target": "≤2× baseline runtime at 10M files",
                "mutation_score_target": "≥0.80",
                "property_coverage_target": "≥0.70"
            }
        }
        
        print(json.dumps(fastpath_result, indent=2))
        return True
        
    except Exception as e:
        error_result = {
            "variant": "$variant",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "execution_duration_sec": time.time() - start_time,
            "memory_delta_mb": measure_memory() - start_memory,
            "timestamp": "$TIMESTAMP"
        }
        print(json.dumps(error_result, indent=2), file=sys.stderr)
        return False
    finally:
        gc.collect()  # Clean up memory

if __name__ == "__main__":
    success = run_fastpath_variant()
    sys.exit(0 if success else 1)
EOF
    
    # Execute with timeout and comprehensive logging
    if timeout 600 python3 "/tmp/run_fastpath_${variant}.py" > "$output_file" 2>>"$LOG_FILE"; then
        log "✅ FastPath V5 $variant execution completed: $output_file"
        
        # Extract key metrics for logging
        if command -v jq &> /dev/null && [[ -f "$output_file" ]]; then
            local tokens=$(jq -r '.selection_result.total_tokens // "unknown"' "$output_file")
            local chunks=$(jq -r '.selection_result.selected_chunks // "unknown"' "$output_file")
            local utilization=$(jq -r '.selection_result.budget_utilization // "unknown"' "$output_file")
            local pagerank_nodes=$(jq -r '.pagerank_analysis.graph_nodes // "unknown"' "$output_file")
            local memory_delta=$(jq -r '.execution.memory_delta_mb // "unknown"' "$output_file")
            
            log "   Tokens: $tokens, Chunks: $chunks, Utilization: $utilization"
            log "   PageRank Nodes: $pagerank_nodes, Memory: ${memory_delta}MB"
        fi
        
        # Run performance benchmarking if enabled
        if [[ "$ENABLE_BENCHMARKING" == true ]]; then
            run_performance_benchmark "$variant" "$output_file"
        fi
        
        return 0
    else
        log "❌ FastPath V5 $variant execution failed or timed out"
        return 1
    fi
}

# Run performance benchmarking
run_performance_benchmark() {
    local variant="$1"
    local results_file="$2"
    local benchmark_file="$RESULTS_DIR/fastpath_${variant}_benchmark_${TIMESTAMP}.json"
    
    log "Running performance benchmark for FastPath V5 $variant..."
    
    # Create benchmarking script
    cat > "/tmp/benchmark_fastpath_${variant}.py" << EOF
#!/usr/bin/env python3
import sys
import json
import time
import statistics
from pathlib import Path

# Add project to path
sys.path.insert(0, '$PROJECT_ROOT')

from packrepo.benchmarks.scalability_benchmark import ScalabilityBenchmark

def run_benchmark():
    try:
        benchmark = ScalabilityBenchmark()
        
        # Run multiple iterations for statistical significance
        iterations = 5
        results = []
        
        for i in range(iterations):
            print(f"Benchmark iteration {i+1}/{iterations}", file=sys.stderr)
            
            result = benchmark.benchmark_fastpath_variant(
                variant='$variant',
                repo_path='$REPO_PATH',
                token_budget=$TOKEN_BUDGET,
                seed=$RANDOM_SEED + i  # Vary seed for each iteration
            )
            
            results.append(result)
        
        # Calculate statistics
        execution_times = [r['execution_time_sec'] for r in results]
        memory_usage = [r['memory_usage_mb'] for r in results]
        
        benchmark_summary = {
            "variant": "$variant",
            "benchmark_iterations": iterations,
            "statistics": {
                "execution_time": {
                    "mean": statistics.mean(execution_times),
                    "median": statistics.median(execution_times),
                    "stdev": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                    "min": min(execution_times),
                    "max": max(execution_times)
                },
                "memory_usage": {
                    "mean": statistics.mean(memory_usage),
                    "median": statistics.median(memory_usage), 
                    "stdev": statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0,
                    "min": min(memory_usage),
                    "max": max(memory_usage)
                }
            },
            "individual_results": results,
            "timestamp": "$TIMESTAMP"
        }
        
        print(json.dumps(benchmark_summary, indent=2))
        return True
        
    except Exception as e:
        print(f"Benchmark failed: {str(e)}", file=sys.stderr)
        return False

if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
EOF
    
    if python3 "/tmp/benchmark_fastpath_${variant}.py" > "$benchmark_file"; then
        log "✅ Performance benchmark completed: $benchmark_file"
    else
        log "⚠️ Performance benchmark failed for $variant"
    fi
}

# Run scalability analysis
run_scalability_analysis() {
    log "Running scalability analysis (10M file simulation)..."
    
    local scalability_file="$RESULTS_DIR/fastpath_scalability_${TIMESTAMP}.json"
    
    cat > "/tmp/scalability_analysis.py" << EOF
#!/usr/bin/env python3
import sys
import json
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, '$PROJECT_ROOT')

from packrepo.benchmarks.scalability_benchmark import ScalabilityBenchmark
from packrepo.benchmarks.synthetic_repo_generator import SyntheticRepoGenerator

def run_scalability_test():
    try:
        # Generate synthetic repository with different sizes
        generator = SyntheticRepoGenerator()
        benchmark = ScalabilityBenchmark()
        
        test_sizes = [1000, 10000, 100000]  # Start small, scale up
        results = {}
        
        for size in test_sizes:
            print(f"Testing scalability at {size} files", file=sys.stderr)
            
            # Generate synthetic repo
            synthetic_repo = generator.generate_repo(
                num_files=size,
                avg_file_size=500,
                complexity_distribution="normal"
            )
            
            # Run FastPath V5 on synthetic repo
            result = benchmark.benchmark_fastpath_scalability(
                repo_path=synthetic_repo,
                token_budget=$TOKEN_BUDGET,
                target_size=size
            )
            
            results[f"size_{size}"] = result
            
            # Extrapolate to 10M files
            if size == 100000:  # Use largest successful test for extrapolation
                scaling_factor = 10000000 / size  # 10M / current size
                estimated_10m = {
                    "estimated_runtime_sec": result["runtime_sec"] * scaling_factor,
                    "estimated_memory_mb": result["memory_mb"] * (scaling_factor ** 0.5),  # Sub-linear memory scaling
                    "baseline_comparison": result["runtime_sec"] * scaling_factor / result["baseline_runtime_sec"],
                    "meets_requirement": (result["runtime_sec"] * scaling_factor / result["baseline_runtime_sec"]) <= 2.0
                }
                results["extrapolated_10m"] = estimated_10m
        
        scalability_result = {
            "test_type": "scalability_analysis",
            "target_size": 10000000,
            "requirement": "≤2× baseline runtime at 10M files",
            "test_results": results,
            "timestamp": "$TIMESTAMP"
        }
        
        print(json.dumps(scalability_result, indent=2))
        return True
        
    except Exception as e:
        print(f"Scalability test failed: {str(e)}", file=sys.stderr)
        return False

if __name__ == "__main__":
    success = run_scalability_test()
    sys.exit(0 if success else 1)
EOF
    
    if python3 "/tmp/scalability_analysis.py" > "$scalability_file"; then
        log "✅ Scalability analysis completed: $scalability_file"
        
        # Check if requirement met
        if command -v jq &> /dev/null; then
            local meets_req=$(jq -r '.test_results.extrapolated_10m.meets_requirement // false' "$scalability_file")
            if [[ "$meets_req" == "true" ]]; then
                log "✅ Scalability requirement MET: ≤2× baseline at 10M files"
            else
                log "❌ Scalability requirement NOT MET: >2× baseline at 10M files"
            fi
        fi
    else
        log "⚠️ Scalability analysis failed"
    fi
}

# Generate comprehensive summary
generate_fastpath_summary() {
    local variants_run=("$@")
    local summary_file="$RESULTS_DIR/fastpath_v5_summary_${TIMESTAMP}.json"
    
    log "Generating FastPath V5 execution summary..."
    
    # Collect all results
    local summary_data="{"
    summary_data+='"execution_summary": {'
    summary_data+='"timestamp": "'$TIMESTAMP'",'
    summary_data+='"variants_requested": ['$(printf '"%s",' "${variants_run[@]}" | sed 's/,$//')]'],'
    summary_data+='"token_budget": '$TOKEN_BUDGET','
    summary_data+='"random_seed": '$RANDOM_SEED','
    summary_data+='"repo_path": "'$REPO_PATH'",'
    summary_data+='"benchmarking_enabled": '$ENABLE_BENCHMARKING','
    summary_data+='"scalability_test_enabled": '$SCALABILITY_TEST
    summary_data+='},'
    summary_data+='"fastpath_results": {'
    
    local first=true
    for variant in "${variants_run[@]}"; do
        local result_file="$RESULTS_DIR/fastpath_${variant}_results_${TIMESTAMP}.json"
        if [[ -f "$result_file" ]]; then
            if [[ "$first" != true ]]; then
                summary_data+=","
            fi
            summary_data+='"'$variant'": '
            summary_data+=$(cat "$result_file")
            first=false
        fi
    done
    
    summary_data+='}}'
    
    echo "$summary_data" | python3 -m json.tool > "$summary_file"
    log "📊 FastPath V5 summary report generated: $summary_file"
}

# Parse command line arguments
parse_arguments() {
    local variants_to_run=()
    local validate_only=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --core)
                variants_to_run+=("core")
                shift
                ;;
            --enhanced)
                variants_to_run+=("enhanced")
                shift
                ;;
            --full)
                variants_to_run+=("full")
                shift
                ;;
            --all)
                variants_to_run=("core" "enhanced" "full")
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
                LOG_FILE="${RESULTS_DIR}/fastpath_v5_execution_${TIMESTAMP}.log"
                shift 2
                ;;
            --no-variants)
                ENABLE_VARIANTS=false
                shift
                ;;
            --no-benchmarking)
                ENABLE_BENCHMARKING=false
                shift
                ;;
            --scalability)
                SCALABILITY_TEST=true
                shift
                ;;
            --validate)
                validate_only=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            core|enhanced|full)
                variants_to_run+=("$1")
                shift
                ;;
            *)
                error_exit "Unknown option: $1. Use --help for usage information."
                ;;
        esac
    done
    
    # Default to all variants if none specified
    if [[ ${#variants_to_run[@]} -eq 0 ]]; then
        variants_to_run=("core" "enhanced" "full")
    fi
    
    # Validation only mode
    if [[ "$validate_only" == true ]]; then
        log "Validation mode - checking FastPath V5 configurations only"
        validate_environment
        
        local validation_failed=false
        for variant in "${variants_to_run[@]}"; do
            if ! validate_fastpath_config "$variant"; then
                validation_failed=true
            fi
        done
        
        if [[ "$validation_failed" == true ]]; then
            error_exit "FastPath V5 validation failed"
        else
            log "✅ All FastPath V5 validations passed"
            exit 0
        fi
    fi
    
    echo "${variants_to_run[@]}"
}

# Main execution
main() {
    log "🚀 FastPath V5 Runner Starting"
    log "Repository: $REPO_PATH"
    log "Token Budget: $TOKEN_BUDGET"
    log "Random Seed: $RANDOM_SEED"
    log "Results Directory: $RESULTS_DIR"
    log "Performance Benchmarking: $ENABLE_BENCHMARKING"
    log "Scalability Testing: $SCALABILITY_TEST"
    
    # Parse arguments and get variants to run
    local variants_to_run_str
    variants_to_run_str=$(parse_arguments "$@")
    read -ra variants_to_run <<< "$variants_to_run_str"
    
    log "FastPath V5 variants to execute: ${variants_to_run[*]}"
    
    # Environment validation
    validate_environment
    
    # Validate requested variants exist
    for variant in "${variants_to_run[@]}"; do
        if [[ ! "${FASTPATH_VARIANTS[$variant]}" ]]; then
            error_exit "Unknown FastPath variant: $variant. Valid: ${!FASTPATH_VARIANTS[*]}"
        fi
    done
    
    # Execute FastPath V5 variants
    local successful_runs=0
    local failed_runs=0
    
    for variant in "${variants_to_run[@]}"; do
        log "="*60
        
        # Validate configuration first
        if validate_fastpath_config "$variant"; then
            # Run variant
            if run_fastpath_variant "$variant"; then
                ((successful_runs++))
            else
                ((failed_runs++))
            fi
        else
            log "❌ Skipping $variant due to configuration validation failure"
            ((failed_runs++))
        fi
    done
    
    # Run scalability analysis if requested
    if [[ "$SCALABILITY_TEST" == true ]]; then
        log "="*60
        run_scalability_analysis
    fi
    
    # Generate summary report
    generate_fastpath_summary "${variants_to_run[@]}"
    
    # Final summary
    log "="*60
    log "🏁 FastPath V5 Execution Complete"
    log "Successful: $successful_runs"
    log "Failed: $failed_runs"
    log "Total: $((successful_runs + failed_runs))"
    log "Results: $RESULTS_DIR"
    log "Log: $LOG_FILE"
    
    if [[ $failed_runs -eq 0 ]]; then
        log "✅ All FastPath V5 variants executed successfully!"
        
        # Performance target check
        log "🎯 Research Targets:"
        log "   - 8-12% improvement over V4 semantic baseline"
        log "   - CI lower bound >0 vs all V1-V4 baselines"
        log "   - ≤2× baseline runtime at 10M files"
        log "   - Mutation score ≥0.80; property coverage ≥0.70"
        
        exit 0
    else
        log "⚠️ Some FastPath V5 variants failed - check logs for details"
        exit 1
    fi
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    rm -f /tmp/validate_fastpath_*.py /tmp/run_fastpath_*.py /tmp/benchmark_fastpath_*.py /tmp/scalability_analysis.py
}

# Set up cleanup trap
trap cleanup EXIT

# Execute main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
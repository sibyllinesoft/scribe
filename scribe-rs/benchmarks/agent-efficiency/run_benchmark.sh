#!/bin/bash
# Agent Token Efficiency Benchmark
# Compares token usage: scribe covering-set vs naive file reading

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIBE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

mkdir -p "$RESULTS_DIR"

# Simple token estimation (chars / 4 is rough approximation for code)
estimate_tokens() {
    local content="$1"
    local chars=$(echo -n "$content" | wc -c)
    echo $((chars / 4))
}

# Count lines in content
count_lines() {
    local content="$1"
    echo -n "$content" | wc -l
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Agent Token Efficiency Benchmark${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Scribe root: $SCRIBE_ROOT"
echo "Results dir: $RESULTS_DIR"
echo ""

# Check if scribe binary exists
if ! command -v scribe &> /dev/null; then
    echo -e "${YELLOW}Building scribe...${NC}"
    (cd "$SCRIBE_ROOT" && cargo build --release)
    SCRIBE_BIN="$SCRIBE_ROOT/target/release/scribe"
else
    SCRIBE_BIN="scribe"
fi

# Benchmark targets (matching targets.json)
declare -A TARGETS
TARGETS["covering_set_compute"]="scribe-selection/src/algorithms/covering_set/mod.rs:compute_covering_set"
TARGETS["token_budget_selection"]="scribe-selection/src/budget/token_budget.rs:apply_token_budget_selection"
TARGETS["centrality_calculate"]="scribe-graph/src/centrality.rs:calculate_centrality"
TARGETS["ast_parse_chunks"]="scribe-selection/src/ast/ast_parser.rs:parse_chunks"
TARGETS["pipeline_analyze_select"]="src/pipeline.rs:analyze_and_select"

# Results arrays
declare -A SCRIBE_TOKENS
declare -A SCRIBE_FILES
declare -A SCRIBE_LINES
declare -A NAIVE_TOKENS
declare -A NAIVE_FILES
declare -A NAIVE_LINES

echo -e "${GREEN}Running benchmarks...${NC}"
echo ""

for target_id in "${!TARGETS[@]}"; do
    query="${TARGETS[$target_id]}"
    echo -e "${YELLOW}Target: $target_id${NC}"
    echo "  Query: $query"

    # Extract file path from query
    target_file="${query%%:*}"
    target_entity="${query##*:}"

    # --- SCRIBE APPROACH ---
    echo -n "  [scribe] Running covering-set... "

    scribe_output=$("$SCRIBE_BIN" --covering-set "$query" --stdout --output-format text 2>/dev/null || echo "ERROR")

    if [[ "$scribe_output" == "ERROR" ]]; then
        echo -e "${RED}FAILED${NC}"
        SCRIBE_TOKENS[$target_id]=0
        SCRIBE_FILES[$target_id]=0
        SCRIBE_LINES[$target_id]=0
    else
        scribe_tokens=$(estimate_tokens "$scribe_output")
        scribe_lines=$(count_lines "$scribe_output")
        # Count files by looking for file markers in output
        scribe_files=$(echo "$scribe_output" | grep -c "^=\+ .* =\+$" 2>/dev/null || echo "0")

        SCRIBE_TOKENS[$target_id]=$scribe_tokens
        SCRIBE_FILES[$target_id]=$scribe_files
        SCRIBE_LINES[$target_id]=$scribe_lines

        echo -e "${GREEN}OK${NC} ($scribe_tokens tokens, $scribe_files files, $scribe_lines lines)"

        # Save scribe output
        echo "$scribe_output" > "$RESULTS_DIR/${target_id}_scribe_${TIMESTAMP}.txt"
    fi

    # --- NAIVE APPROACH ---
    # Simulate what an agent would do: read target file, grep for imports,
    # read imported files, repeat
    echo -n "  [naive] Simulating file reads... "

    naive_content=""
    naive_files=0

    # Start with target file
    if [[ -f "$SCRIBE_ROOT/$target_file" ]]; then
        naive_content=$(cat "$SCRIBE_ROOT/$target_file")
        naive_files=1

        # Extract use/import statements and find those files
        imports=$(grep -E "^use (scribe_|crate::)" "$SCRIBE_ROOT/$target_file" 2>/dev/null | head -20 || true)

        # For each import, try to find and read the source file
        while IFS= read -r import_line; do
            if [[ -z "$import_line" ]]; then continue; fi

            # Extract crate name from use statement
            crate_name=$(echo "$import_line" | sed -E 's/use (scribe_[a-z_]+).*/\1/' | sed 's/_/-/g')

            # Try to find lib.rs or mod.rs for the crate
            for candidate in "$SCRIBE_ROOT/$crate_name/src/lib.rs" "$SCRIBE_ROOT/$crate_name/src/mod.rs"; do
                if [[ -f "$candidate" ]]; then
                    naive_content+=$'\n'"$(cat "$candidate")"
                    ((naive_files++))
                    break
                fi
            done
        done <<< "$imports"
    fi

    naive_tokens=$(estimate_tokens "$naive_content")
    naive_lines=$(count_lines "$naive_content")

    NAIVE_TOKENS[$target_id]=$naive_tokens
    NAIVE_FILES[$target_id]=$naive_files
    NAIVE_LINES[$target_id]=$naive_lines

    echo -e "${GREEN}OK${NC} ($naive_tokens tokens, $naive_files files, $naive_lines lines)"

    # Save naive output
    echo "$naive_content" > "$RESULTS_DIR/${target_id}_naive_${TIMESTAMP}.txt"

    echo ""
done

# --- GENERATE REPORT ---
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Results Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

report_file="$RESULTS_DIR/report_${TIMESTAMP}.md"

cat > "$report_file" << EOF
# Agent Token Efficiency Benchmark Results

**Date:** $(date)
**Scribe Version:** $($SCRIBE_BIN --version 2>/dev/null || echo "unknown")

## Summary

| Target | Scribe Tokens | Naive Tokens | Savings | Scribe Files | Naive Files |
|--------|---------------|--------------|---------|--------------|-------------|
EOF

total_scribe=0
total_naive=0

for target_id in "${!TARGETS[@]}"; do
    s_tokens=${SCRIBE_TOKENS[$target_id]}
    n_tokens=${NAIVE_TOKENS[$target_id]}
    s_files=${SCRIBE_FILES[$target_id]}
    n_files=${NAIVE_FILES[$target_id]}

    if [[ $n_tokens -gt 0 ]]; then
        savings=$(echo "scale=1; (1 - $s_tokens / $n_tokens) * 100" | bc)
        ratio=$(echo "scale=1; $n_tokens / $s_tokens" | bc 2>/dev/null || echo "N/A")
    else
        savings="N/A"
        ratio="N/A"
    fi

    echo "| $target_id | $s_tokens | $n_tokens | ${savings}% | $s_files | $n_files |" >> "$report_file"

    total_scribe=$((total_scribe + s_tokens))
    total_naive=$((total_naive + n_tokens))
done

if [[ $total_naive -gt 0 ]]; then
    total_savings=$(echo "scale=1; (1 - $total_scribe / $total_naive) * 100" | bc)
    total_ratio=$(echo "scale=1; $total_naive / $total_scribe" | bc)
else
    total_savings="N/A"
    total_ratio="N/A"
fi

cat >> "$report_file" << EOF
| **TOTAL** | **$total_scribe** | **$total_naive** | **${total_savings}%** | - | - |

## Key Findings

- **Total token savings:** ${total_savings}%
- **Efficiency ratio:** ${total_ratio}x fewer tokens with scribe

## Methodology

- **Scribe approach:** Single \`scribe --covering-set\` call returning only relevant code
- **Naive approach:** Read target file + grep for imports + read imported crate lib.rs files
- **Token estimation:** characters / 4 (rough approximation for code)

## Notes

The naive approach is actually *optimistic* because a real agent would:
1. Make multiple iterative grep/read calls (more tool call overhead)
2. Read more files as it discovers transitive dependencies
3. Potentially read irrelevant files due to ambiguous grep matches

EOF

echo "Report saved to: $report_file"
echo ""
cat "$report_file"

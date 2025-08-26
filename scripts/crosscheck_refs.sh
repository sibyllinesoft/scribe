#!/bin/bash

# FastPath V5 Citation Verification Script
# Automated citation cross-checking for ICSE 2025 submission
# Usage: ./crosscheck_refs.sh [latex_file] [bib_file]

set -euo pipefail

# Configuration
LATEX_FILE="${1:-fastpath_v5_icse2025_research_paper.tex}"
BIB_FILE="${2:-paper/refs.bib}"
REPORT_FILE="citation_verification_$(date +%Y%m%d_%H%M%S).md"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${2:-$NC}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Check if required files exist
check_files() {
    if [[ ! -f "$LATEX_FILE" ]]; then
        log "ERROR: LaTeX file '$LATEX_FILE' not found!" "$RED"
        exit 1
    fi
    
    if [[ ! -f "$BIB_FILE" ]]; then
        log "ERROR: Bibliography file '$BIB_FILE' not found!" "$RED"
        exit 1
    fi
}

# Extract citation keys from LaTeX file
extract_latex_citations() {
    log "Extracting citations from LaTeX file..." "$BLUE"
    grep -o '\\cite{[^}]*}' "$LATEX_FILE" | \
        sed 's/\\cite{//g' | \
        sed 's/}//g' | \
        sort | uniq > /tmp/latex_citations.txt
    
    local count=$(wc -l < /tmp/latex_citations.txt)
    log "Found $count unique citations in LaTeX file" "$GREEN"
}

# Extract citation keys from .bib file  
extract_bib_citations() {
    log "Extracting citations from bibliography file..." "$BLUE"
    grep -o '@[a-zA-Z]*{[^,]*,' "$BIB_FILE" | \
        sed 's/@[^{]*{//g' | \
        sed 's/,//g' | \
        sort > /tmp/bib_citations.txt
        
    local count=$(wc -l < /tmp/bib_citations.txt)
    log "Found $count entries in bibliography file" "$GREEN"
}

# Compare citations and find mismatches
compare_citations() {
    log "Cross-referencing citations..." "$BLUE"
    
    # Missing from .bib file
    comm -23 /tmp/latex_citations.txt /tmp/bib_citations.txt > /tmp/missing_in_bib.txt
    local missing_count=$(wc -l < /tmp/missing_in_bib.txt)
    
    # Unused in LaTeX
    comm -13 /tmp/latex_citations.txt /tmp/bib_citations.txt > /tmp/unused_in_latex.txt
    local unused_count=$(wc -l < /tmp/unused_in_latex.txt)
    
    log "Citations missing from .bib file: $missing_count" "$YELLOW"
    log "Bibliography entries unused in LaTeX: $unused_count" "$YELLOW"
}

# Check DOI accessibility
check_dois() {
    log "Checking DOI accessibility..." "$BLUE"
    
    # Extract DOIs from .bib file
    grep -o 'doi={[^}]*}' "$BIB_FILE" | \
        sed 's/doi={//g' | \
        sed 's/}//g' > /tmp/dois.txt
    
    local doi_count=$(wc -l < /tmp/dois.txt)
    log "Found $doi_count DOIs to check" "$GREEN"
    
    local accessible=0
    local inaccessible=0
    
    while IFS= read -r doi; do
        if curl -s --head "https://doi.org/$doi" | grep -q "200 OK"; then
            accessible=$((accessible + 1))
        else
            echo "$doi" >> /tmp/inaccessible_dois.txt
            inaccessible=$((inaccessible + 1))
        fi
    done < /tmp/dois.txt
    
    log "DOIs accessible: $accessible" "$GREEN"
    log "DOIs inaccessible: $inaccessible" "$RED"
}

# Validate bibliography format
validate_bib_format() {
    log "Validating bibliography format..." "$BLUE"
    
    local errors=0
    
    # Check for required fields
    while IFS= read -r citation; do
        if ! grep -A 20 "^@.*{$citation," "$BIB_FILE" | grep -q "title="; then
            echo "Missing title: $citation" >> /tmp/format_errors.txt
            errors=$((errors + 1))
        fi
        
        if ! grep -A 20 "^@.*{$citation," "$BIB_FILE" | grep -q "author="; then
            echo "Missing author: $citation" >> /tmp/format_errors.txt
            errors=$((errors + 1))
        fi
        
        if ! grep -A 20 "^@.*{$citation," "$BIB_FILE" | grep -q "year="; then
            echo "Missing year: $citation" >> /tmp/format_errors.txt
            errors=$((errors + 1))
        fi
    done < /tmp/bib_citations.txt
    
    log "Bibliography format errors: $errors" "$YELLOW"
}

# Generate comprehensive report
generate_report() {
    log "Generating verification report..." "$BLUE"
    
    cat > "$REPORT_FILE" << EOF
# Citation Verification Report
**Generated**: $(date)  
**LaTeX File**: $LATEX_FILE  
**Bibliography File**: $BIB_FILE  

## Summary
- Citations in LaTeX: $(wc -l < /tmp/latex_citations.txt)
- Entries in .bib file: $(wc -l < /tmp/bib_citations.txt)  
- Missing from .bib: $(wc -l < /tmp/missing_in_bib.txt)
- Unused in LaTeX: $(wc -l < /tmp/unused_in_latex.txt)
- DOIs checked: $(wc -l < /tmp/dois.txt)
- Format errors: $(wc -l < /tmp/format_errors.txt 2>/dev/null || echo "0")

## Citations Missing from Bibliography
EOF

    if [[ -s /tmp/missing_in_bib.txt ]]; then
        echo "❌ **CRITICAL ISSUES FOUND**" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        while IFS= read -r citation; do
            echo "- \`$citation\`" >> "$REPORT_FILE"
        done < /tmp/missing_in_bib.txt
    else
        echo "✅ All citations have corresponding bibliography entries" >> "$REPORT_FILE"
    fi
    
    cat >> "$REPORT_FILE" << EOF

## Unused Bibliography Entries
EOF

    if [[ -s /tmp/unused_in_latex.txt ]]; then
        echo "" >> "$REPORT_FILE"
        while IFS= read -r citation; do
            echo "- \`$citation\`" >> "$REPORT_FILE"
        done < /tmp/unused_in_latex.txt
    else
        echo "✅ All bibliography entries are used in LaTeX" >> "$REPORT_FILE"
    fi

    if [[ -f /tmp/inaccessible_dois.txt && -s /tmp/inaccessible_dois.txt ]]; then
        cat >> "$REPORT_FILE" << EOF

## Inaccessible DOIs
❌ **The following DOIs are not accessible:**
EOF
        while IFS= read -r doi; do
            echo "- $doi" >> "$REPORT_FILE"
        done < /tmp/inaccessible_dois.txt
    fi

    if [[ -f /tmp/format_errors.txt && -s /tmp/format_errors.txt ]]; then
        cat >> "$REPORT_FILE" << EOF

## Format Errors
❌ **Bibliography format issues found:**
EOF
        cat /tmp/format_errors.txt >> "$REPORT_FILE"
    fi

    log "Report generated: $REPORT_FILE" "$GREEN"
}

# Cleanup function
cleanup() {
    rm -f /tmp/latex_citations.txt /tmp/bib_citations.txt
    rm -f /tmp/missing_in_bib.txt /tmp/unused_in_latex.txt
    rm -f /tmp/dois.txt /tmp/inaccessible_dois.txt
    rm -f /tmp/format_errors.txt
}

# Main execution
main() {
    log "Starting citation verification..." "$GREEN"
    
    check_files
    extract_latex_citations
    extract_bib_citations  
    compare_citations
    check_dois
    validate_bib_format
    generate_report
    
    # Summary
    local missing=$(wc -l < /tmp/missing_in_bib.txt)
    local errors=$(wc -l < /tmp/format_errors.txt 2>/dev/null || echo "0")
    
    if [[ $missing -eq 0 && $errors -eq 0 ]]; then
        log "✅ VERIFICATION PASSED - No critical issues found!" "$GREEN"
    else
        log "❌ VERIFICATION FAILED - Issues found, check report!" "$RED"
        exit 1
    fi
    
    cleanup
    log "Verification complete! Report: $REPORT_FILE" "$BLUE"
}

# Set up trap for cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"
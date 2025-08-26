#!/bin/bash
# Compile FastPath V5 ICSE 2025 Paper

echo "🚀 Compiling FastPath V5 ICSE 2025 Paper..."

# Set working directory
cd "$(dirname "$0")"

# Check if required files exist
if [ ! -f "fastpath_v5_icse2025_paper.tex" ]; then
    echo "❌ Error: fastpath_v5_icse2025_paper.tex not found"
    exit 1
fi

if [ ! -f "references.bib" ]; then
    echo "❌ Error: references.bib not found"
    exit 1
fi

# Check if figures directory exists
if [ ! -d "figures" ]; then
    echo "📊 Generating figures..."
    python3 figures/generate_figures.py
fi

# Compile LaTeX document
echo "📝 Compiling LaTeX document..."

# First pass
pdflatex -interaction=nonstopmode fastpath_v5_icse2025_paper.tex > compile.log 2>&1
if [ $? -ne 0 ]; then
    echo "❌ First LaTeX compilation failed. Check compile.log for errors."
    exit 1
fi

# Run bibtex for references
echo "📚 Processing bibliography..."
bibtex fastpath_v5_icse2025_paper > bibtex.log 2>&1
if [ $? -ne 0 ]; then
    echo "⚠️  BibTeX warning. Check bibtex.log for details."
fi

# Second pass for cross-references
pdflatex -interaction=nonstopmode fastpath_v5_icse2025_paper.tex >> compile.log 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Second LaTeX compilation failed. Check compile.log for errors."
    exit 1
fi

# Third pass to resolve all references
pdflatex -interaction=nonstopmode fastpath_v5_icse2025_paper.tex >> compile.log 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Final LaTeX compilation failed. Check compile.log for errors."
    exit 1
fi

# Clean up auxiliary files (optional)
echo "🧹 Cleaning auxiliary files..."
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot

# Check if PDF was generated
if [ -f "fastpath_v5_icse2025_paper.pdf" ]; then
    echo "✅ Paper compiled successfully!"
    echo "📄 Output: fastpath_v5_icse2025_paper.pdf"
    
    # Get PDF info
    if command -v pdfinfo >/dev/null 2>&1; then
        pages=$(pdfinfo fastpath_v5_icse2025_paper.pdf | grep "Pages:" | awk '{print $2}')
        echo "📊 Pages: $pages"
        
        # Check ICSE page limit (8 pages + references)
        if [ "$pages" -le 12 ]; then
            echo "✅ Page count within ICSE limits"
        else
            echo "⚠️  Page count may exceed ICSE limits ($pages pages)"
        fi
    fi
    
    # Get file size
    size=$(du -h fastpath_v5_icse2025_paper.pdf | cut -f1)
    echo "💾 File size: $size"
    
else
    echo "❌ PDF generation failed"
    exit 1
fi

echo ""
echo "🎯 FastPath V5 ICSE 2025 Paper - Compilation Complete!"
echo "📁 Location: $(pwd)/fastpath_v5_icse2025_paper.pdf"
echo ""
echo "📋 Submission Checklist:"
echo "  ✅ Paper compiled successfully"
echo "  ✅ Bibliography processed" 
echo "  ✅ Figures included"
echo "  ✅ Page count checked"
echo ""
echo "🚀 Ready for ICSE 2025 submission!"
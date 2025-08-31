#!/usr/bin/env python3
"""
Update LaTeX paper with generated tables and figures.

Patches LaTeX documents by:
- Replacing placeholder values with actual results
- Updating table references with generated content  
- Updating figure paths and captions
- Preserving document structure and formatting
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LaTeXPatcher:
    """Update LaTeX documents with evaluation results."""
    
    def __init__(self):
        self.placeholder_pattern = r'\\placeholder\{([^}]+)\}'
        self.table_pattern = r'\\input\{([^}]+_table[^}]*)\}'
        self.figure_pattern = r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
        
    def load_tex_file(self, tex_file: Path) -> str:
        """Load LaTeX file content."""
        with open(tex_file, 'r', encoding='utf-8') as f:
            return f.read()
            
    def load_tables(self, tables_file: Path) -> str:
        """Load generated LaTeX tables."""
        with open(tables_file, 'r', encoding='utf-8') as f:
            return f.read()
            
    def update_placeholders(self, content: str, results: Dict[str, Any]) -> str:
        """Update placeholder values with actual results."""
        
        def replace_placeholder(match):
            placeholder = match.group(1)
            
            # Define placeholder mappings
            replacements = {
                'MAIN_IMPROVEMENT': self._format_improvement(results),
                'TARGET_IMPROVEMENT': '13.0',
                'CONFIDENCE_LEVEL': '95',
                'BOOTSTRAP_ITERATIONS': '10000',
                'EVALUATION_SEEDS': '100',
                'TOKEN_BUDGETS': '50k, 120k, 200k',
                'SYSTEMS_EVALUATED': self._format_systems(results),
                'SIGNIFICANCE_THRESHOLD': '0.05'
            }
            
            return replacements.get(placeholder, f'\\texttt{{{placeholder}}}')
            
        return re.sub(self.placeholder_pattern, replace_placeholder, content)
        
    def update_table_references(self, content: str, tables_content: str) -> str:
        """Update table input references with generated tables."""
        
        # Extract individual tables from tables content
        tables = self._extract_tables(tables_content)
        
        def replace_table_input(match):
            table_file = match.group(1)
            
            # Map table files to generated content
            if 'main_results' in table_file:
                return tables.get('main_results', match.group(0))
            elif 'category' in table_file:
                return tables.get('category_breakdown', match.group(0))
            elif 'negative' in table_file:
                return tables.get('negative_controls', match.group(0))
            elif 'baselines' in table_file:
                return tables.get('ir_baselines', match.group(0))
            else:
                return match.group(0)
                
        return re.sub(self.table_pattern, replace_table_input, content)
        
    def update_figure_references(self, content: str, figures_dir: Path) -> str:
        """Update figure references with generated figures."""
        
        def replace_figure_path(match):
            figure_file = match.group(1)
            
            # Map figure names to generated files
            figure_mappings = {
                'performance_comparison': 'performance_comparison.pdf',
                'confidence_intervals': 'confidence_intervals.pdf', 
                'category_breakdown': 'category_breakdown.pdf',
                'effect_sizes': 'effect_sizes.pdf'
            }
            
            # Check if this is a figure we generated
            for key, generated_file in figure_mappings.items():
                if key in figure_file:
                    new_path = figures_dir / generated_file
                    if new_path.exists():
                        return str(new_path)
                        
            return figure_file
            
        return re.sub(self.figure_pattern, 
                     lambda m: f'\\includegraphics{{{replace_figure_path(m)}}}', 
                     content)
        
    def _extract_tables(self, tables_content: str) -> Dict[str, str]:
        """Extract individual tables from consolidated content."""
        tables = {}
        
        # Split by table boundaries
        table_sections = re.split(r'(?=\\begin\{table\})', tables_content)
        
        for section in table_sections:
            if not section.strip():
                continue
                
            # Identify table type from label or caption
            if 'main_results' in section or 'Main Results' in section:
                tables['main_results'] = section.strip()
            elif 'category' in section or 'Category' in section:
                tables['category_breakdown'] = section.strip()
            elif 'negative' in section or 'Negative' in section:
                tables['negative_controls'] = section.strip()
            elif 'baselines' in section or 'Baselines' in section:
                tables['ir_baselines'] = section.strip()
                
        return tables
        
    def _format_improvement(self, results: Dict[str, Any]) -> str:
        """Format main improvement percentage."""
        # Extract from results if available
        if 'system_analyses' in results:
            for system, analysis in results['system_analyses'].items():
                if system != 'baseline' and 'overall_improvement_pct' in analysis:
                    improvement = analysis['overall_improvement_pct']
                    return f"{improvement:+.1f}\\%"
                    
        return "+XX.X\\%"  # Placeholder if not available
        
    def _format_systems(self, results: Dict[str, Any]) -> str:
        """Format list of evaluated systems."""
        if 'system_analyses' in results:
            systems = list(results['system_analyses'].keys())
            return ", ".join(s.upper() for s in systems if s != 'baseline')
            
        return "V1, V2, V3, V4, V5"
        
    def patch_document(
        self,
        tex_file: Path,
        tables_file: Path,
        figures_dir: Path,
        results: Optional[Dict[str, Any]] = None
    ) -> str:
        """Patch LaTeX document with generated content."""
        
        # Load content
        content = self.load_tex_file(tex_file)
        tables_content = self.load_tables(tables_file)
        
        logger.info(f"Patching LaTeX document: {tex_file}")
        
        # Apply updates
        if results:
            content = self.update_placeholders(content, results)
            
        content = self.update_table_references(content, tables_content)
        content = self.update_figure_references(content, figures_dir)
        
        return content
        
    def save_patched_document(
        self,
        original_tex: Path,
        patched_content: str,
        output_tex: Optional[Path] = None
    ):
        """Save patched document."""
        
        if output_tex is None:
            output_tex = original_tex
            
        # Create backup of original
        if output_tex == original_tex:
            backup_file = original_tex.with_suffix('.tex.backup')
            original_tex.rename(backup_file)
            logger.info(f"Created backup: {backup_file}")
            
        # Save patched content
        with open(output_tex, 'w', encoding='utf-8') as f:
            f.write(patched_content)
            
        logger.info(f"Saved patched document: {output_tex}")


def main():
    parser = argparse.ArgumentParser(
        description="Update LaTeX paper with generated tables and figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python patch_tex.py --tex paper/fastpath.tex --tables paper/tables.tex --figdir paper/figures --results artifacts/analysis.json
        """
    )
    
    parser.add_argument('--tex', type=str, required=True,
                        help='LaTeX document to patch')
    parser.add_argument('--tables', type=str, required=True,
                        help='Generated LaTeX tables file')
    parser.add_argument('--figdir', type=str, required=True,
                        help='Directory containing generated figures')
    parser.add_argument('--results', type=str,
                        help='Analysis results JSON file (optional)')
    parser.add_argument('--output', type=str,
                        help='Output LaTeX file (default: overwrite input)')
    parser.add_argument('--backup', action='store_true', default=True,
                        help='Create backup of original file')
    
    args = parser.parse_args()
    
    # Load results if provided
    results = None
    if args.results:
        with open(args.results, 'r') as f:
            results = json.load(f)
            
    # Create patcher
    patcher = LaTeXPatcher()
    
    # Patch document
    patched_content = patcher.patch_document(
        tex_file=Path(args.tex),
        tables_file=Path(args.tables),
        figures_dir=Path(args.figdir),
        results=results
    )
    
    # Save patched document
    output_file = Path(args.output) if args.output else Path(args.tex)
    patcher.save_patched_document(
        original_tex=Path(args.tex),
        patched_content=patched_content,
        output_tex=output_file
    )
    
    print(f"\nLaTeX document patched successfully!")
    print(f"Input: {args.tex}")
    print(f"Output: {output_file}")
    print(f"Tables: {args.tables}")
    print(f"Figures: {args.figdir}")
    
    if args.backup and output_file == Path(args.tex):
        backup_file = Path(args.tex).with_suffix('.tex.backup')
        print(f"Backup: {backup_file}")


if __name__ == "__main__":
    main()
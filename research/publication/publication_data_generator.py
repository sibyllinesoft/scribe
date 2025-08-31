#!/usr/bin/env python3
"""
Publication Data Generator for FastPath Research
===============================================

Generates publication-ready outputs:
- LaTeX tables with statistical formatting
- High-quality statistical plots with confidence intervals
- IEEE/ACM conference format compliance
- Performance comparison visualizations
- Research paper figures and tables

All outputs are production-ready for academic publication.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Set publication-quality style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'text.usetex': False,  # Set to True if LaTeX is available
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Publication color palette
PUBLICATION_COLORS = {
    'fastpath_v3': '#1f77b4',      # Blue
    'fastpath_v2': '#ff7f0e',      # Orange  
    'fastpath_v1': '#2ca02c',      # Green
    'bm25': '#d62728',             # Red
    'naive_tfidf': '#9467bd',      # Purple
    'random': '#8c564b'            # Brown
}


class PublicationDataGenerator:
    """
    Generates publication-ready tables, figures, and data exports.
    
    Creates IEEE/ACM conference quality outputs suitable for
    research paper inclusion and peer review.
    """
    
    def __init__(self, output_directory: str):
        """Initialize with output directory."""
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'tables').mkdir(exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'data').mkdir(exist_ok=True)
        (self.output_dir / 'latex').mkdir(exist_ok=True)
    
    def generate_all_outputs(
        self,
        raw_measurements: List[Dict[str, Any]],
        statistical_results: Dict[str, Any],
        config: Any
    ) -> Dict[str, List[str]]:
        """Generate all publication outputs."""
        
        generated_files = {
            'tables': [],
            'figures': [],
            'data_files': [],
            'latex_files': []
        }
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(raw_measurements)
        successful_df = df[df['success'] == True].copy()
        
        if len(successful_df) == 0:
            raise ValueError("No successful measurements to generate outputs from")
        
        # Generate performance comparison table
        table_file = self.generate_performance_table(statistical_results['system_summaries'])
        generated_files['tables'].append(table_file)
        
        # Generate statistical significance table
        sig_table_file = self.generate_significance_table(statistical_results['significance_tests'])
        generated_files['tables'].append(sig_table_file)
        
        # Generate effect sizes table
        effect_table_file = self.generate_effect_sizes_table(statistical_results['effect_sizes'])
        generated_files['tables'].append(effect_table_file)
        
        # Generate performance comparison plots
        comparison_plot = self.generate_performance_comparison_plot(successful_df)
        generated_files['figures'].append(comparison_plot)
        
        # Generate execution time analysis
        time_plot = self.generate_execution_time_analysis(successful_df)
        generated_files['figures'].append(time_plot)
        
        # Generate QA accuracy analysis
        qa_plot = self.generate_qa_accuracy_analysis(successful_df)
        generated_files['figures'].append(qa_plot)
        
        # Generate memory usage analysis
        memory_plot = self.generate_memory_usage_analysis(successful_df)
        generated_files['figures'].append(memory_plot)
        
        # Generate bootstrap confidence intervals plot
        bootstrap_plot = self.generate_bootstrap_intervals_plot(statistical_results['bootstrap_intervals'])
        generated_files['figures'].append(bootstrap_plot)
        
        # Generate system comparison heatmap
        heatmap_plot = self.generate_system_comparison_heatmap(statistical_results['system_summaries'])
        generated_files['figures'].append(heatmap_plot)
        
        # Export raw data for transparency
        data_files = self.export_research_data(raw_measurements, statistical_results)
        generated_files['data_files'].extend(data_files)
        
        # Generate LaTeX document template
        latex_file = self.generate_latex_document_template(generated_files, config)
        generated_files['latex_files'].append(latex_file)
        
        # Generate results summary
        summary_file = self.generate_results_summary(statistical_results, config)
        generated_files['data_files'].append(summary_file)
        
        return generated_files
    
    def generate_performance_table(self, system_summaries: Dict[str, Dict[str, Any]]) -> str:
        """Generate main performance comparison table in LaTeX."""
        
        # Define metrics to include in table
        metrics = {
            'qa_accuracy': {'name': 'QA Accuracy', 'format': '%.3f', 'higher_better': True},
            'qa_f1_score': {'name': 'F1 Score', 'format': '%.3f', 'higher_better': True},
            'execution_time_seconds': {'name': 'Exec. Time (s)', 'format': '%.2f', 'higher_better': False},
            'memory_usage_bytes': {'name': 'Memory (MB)', 'format': '%.1f', 'higher_better': False}
        }
        
        # Start LaTeX table
        latex_content = []
        latex_content.append("\\begin{table*}[t]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Performance Comparison of Retrieval Systems}")
        latex_content.append("\\label{tab:performance_comparison}")
        latex_content.append("\\begin{tabular}{l" + "c" * len(metrics) + "}")
        latex_content.append("\\toprule")
        
        # Header
        header = ["System"] + [metrics[m]['name'] for m in metrics.keys()]
        latex_content.append(" & ".join(header) + " \\\\")
        latex_content.append("\\midrule")
        
        # Data rows
        for system, summary in system_summaries.items():
            row_data = [self._format_system_name(system)]
            
            for metric, metric_info in metrics.items():
                if metric in summary:
                    mean_val = summary[metric]['mean']
                    std_val = summary[metric]['std']
                    
                    # Convert bytes to MB for memory
                    if metric == 'memory_usage_bytes':
                        mean_val /= (1024 * 1024)
                        std_val /= (1024 * 1024)
                    
                    # Format with standard deviation
                    formatted_val = f"{mean_val:{metric_info['format']}} $\\pm$ {std_val:{metric_info['format']}}"
                    
                    # Bold best performing systems
                    if self._is_best_performance(system, metric, system_summaries, metric_info['higher_better']):
                        formatted_val = f"\\textbf{{{formatted_val}}}"
                    
                    row_data.append(formatted_val)
                else:
                    row_data.append("--")
            
            latex_content.append(" & ".join(row_data) + " \\\\")
        
        # End table
        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table*}")
        
        # Save to file
        table_file = self.output_dir / 'latex' / 'performance_table.tex'
        with open(table_file, 'w') as f:
            f.write('\n'.join(latex_content))
        
        return str(table_file)
    
    def generate_significance_table(self, significance_tests: Dict[str, Any]) -> str:
        """Generate statistical significance table."""
        
        latex_content = []
        latex_content.append("\\begin{table}[t]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Statistical Significance Tests}")
        latex_content.append("\\label{tab:significance_tests}")
        latex_content.append("\\begin{tabular}{lcc}")
        latex_content.append("\\toprule")
        latex_content.append("Test & Statistic & p-value \\\\")
        latex_content.append("\\midrule")
        
        for test_name, test_data in significance_tests.items():
            if isinstance(test_data, dict) and 'statistic' in test_data:
                statistic = test_data['statistic']
                p_value = test_data['p_value']
                
                # Format test name
                formatted_name = test_name.replace('_', ' ').title()
                
                # Format p-value with significance indicators
                if p_value < 0.001:
                    p_str = "< 0.001***"
                elif p_value < 0.01:
                    p_str = f"{p_value:.3f}**"
                elif p_value < 0.05:
                    p_str = f"{p_value:.3f}*"
                else:
                    p_str = f"{p_value:.3f}"
                
                latex_content.append(f"{formatted_name} & {statistic:.3f} & {p_str} \\\\")
        
        latex_content.append("\\bottomrule")
        latex_content.append("\\multicolumn{3}{l}{\\footnotesize *p < 0.05, **p < 0.01, ***p < 0.001} \\\\")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        
        table_file = self.output_dir / 'latex' / 'significance_table.tex'
        with open(table_file, 'w') as f:
            f.write('\n'.join(latex_content))
        
        return str(table_file)
    
    def generate_effect_sizes_table(self, effect_sizes: Dict[str, Any]) -> str:
        """Generate effect sizes table."""
        
        latex_content = []
        latex_content.append("\\begin{table}[t]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Effect Sizes (Cohen's d) for FastPath vs Baselines}")
        latex_content.append("\\label{tab:effect_sizes}")
        latex_content.append("\\begin{tabular}{llcc}")
        latex_content.append("\\toprule")
        latex_content.append("FastPath Version & Baseline & Metric & Effect Size \\\\")
        latex_content.append("\\midrule")
        
        for fastpath_system, comparisons in effect_sizes.items():
            if isinstance(comparisons, dict):
                for comparison, metrics in comparisons.items():
                    baseline = comparison.replace('vs_', '')
                    
                    for metric, effect_data in metrics.items():
                        if isinstance(effect_data, dict) and 'value' in effect_data:
                            effect_value = effect_data['value']
                            magnitude = effect_data.get('magnitude', 'unknown')
                            
                            # Format system names
                            fastpath_name = self._format_system_name(fastpath_system)
                            baseline_name = self._format_system_name(baseline)
                            metric_name = metric.replace('_', ' ').title()
                            
                            # Color code by magnitude
                            if magnitude == 'large':
                                effect_str = f"\\textbf{{{effect_value:.3f}}}"
                            elif magnitude == 'medium':
                                effect_str = f"\\textit{{{effect_value:.3f}}}"
                            else:
                                effect_str = f"{effect_value:.3f}"
                            
                            latex_content.append(f"{fastpath_name} & {baseline_name} & {metric_name} & {effect_str} \\\\")
        
        latex_content.append("\\bottomrule")
        latex_content.append("\\multicolumn{4}{l}{\\footnotesize Bold: large effect ($|d| > 0.8$), Italic: medium effect ($|d| > 0.5$)} \\\\")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        
        table_file = self.output_dir / 'latex' / 'effect_sizes_table.tex'
        with open(table_file, 'w') as f:
            f.write('\n'.join(latex_content))
        
        return str(table_file)
    
    def generate_performance_comparison_plot(self, df: pd.DataFrame) -> str:
        """Generate performance comparison plot."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # QA Accuracy comparison
        self._create_boxplot(df, 'qa_accuracy', 'QA Accuracy', ax1)
        
        # Execution time comparison
        self._create_boxplot(df, 'execution_time_seconds', 'Execution Time (s)', ax2, log_scale=True)
        
        # F1 Score comparison
        self._create_boxplot(df, 'qa_f1_score', 'F1 Score', ax3)
        
        # Memory usage comparison
        df_memory = df.copy()
        df_memory['memory_usage_mb'] = df_memory['memory_usage_bytes'] / (1024 * 1024)
        self._create_boxplot(df_memory, 'memory_usage_mb', 'Memory Usage (MB)', ax4, log_scale=True)
        
        plt.tight_layout()
        
        plot_file = self.output_dir / 'figures' / 'performance_comparison.pdf'
        plt.savefig(plot_file, format='pdf')
        plt.close()
        
        return str(plot_file)
    
    def generate_execution_time_analysis(self, df: pd.DataFrame) -> str:
        """Generate detailed execution time analysis."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Box plot with statistical annotations
        systems = df['system_name'].unique()
        execution_times = []
        labels = []
        
        for system in systems:
            times = df[df['system_name'] == system]['execution_time_seconds'].dropna()
            if len(times) > 0:
                execution_times.append(times)
                labels.append(self._format_system_name(system))
        
        # Create box plot
        bp = ax1.boxplot(execution_times, labels=labels, patch_artist=True)
        
        # Color boxes
        for patch, system in zip(bp['boxes'], systems):
            patch.set_facecolor(PUBLICATION_COLORS.get(system, '#cccccc'))
            patch.set_alpha(0.7)
        
        ax1.set_title('Execution Time Distribution')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_yscale('log')
        ax1.tick_params(axis='x', rotation=45)
        
        # Performance improvement plot
        baseline_system = 'random'  # Use random as baseline
        if baseline_system in df['system_name'].values:
            baseline_mean = df[df['system_name'] == baseline_system]['execution_time_seconds'].mean()
            
            improvements = []
            system_names = []
            
            for system in systems:
                if system != baseline_system:
                    system_mean = df[df['system_name'] == system]['execution_time_seconds'].mean()
                    improvement = ((baseline_mean - system_mean) / baseline_mean) * 100
                    improvements.append(improvement)
                    system_names.append(self._format_system_name(system))
            
            bars = ax2.bar(system_names, improvements, 
                          color=[PUBLICATION_COLORS.get(s, '#cccccc') for s in systems if s != baseline_system])
            ax2.set_title('Speed Improvement vs Random Baseline')
            ax2.set_ylabel('Improvement (%)')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, improvement in zip(bars, improvements):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                        f'{improvement:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        
        plot_file = self.output_dir / 'figures' / 'execution_time_analysis.pdf'
        plt.savefig(plot_file, format='pdf')
        plt.close()
        
        return str(plot_file)
    
    def generate_qa_accuracy_analysis(self, df: pd.DataFrame) -> str:
        """Generate QA accuracy analysis with confidence intervals."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # System performance with error bars
        systems = sorted(df['system_name'].unique())
        means = []
        stds = []
        colors = []
        
        for system in systems:
            system_data = df[df['system_name'] == system]['qa_accuracy'].dropna()
            if len(system_data) > 0:
                means.append(system_data.mean())
                stds.append(system_data.std())
                colors.append(PUBLICATION_COLORS.get(system, '#cccccc'))
        
        x_pos = range(len(systems))
        bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, 
                       color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('System')
        ax1.set_ylabel('QA Accuracy')
        ax1.set_title('QA Accuracy by System (Mean ± SD)')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([self._format_system_name(s) for s in systems], rotation=45)
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom')
        
        # Accuracy distribution by repository type
        if 'repository_type' in df.columns:
            repo_types = df['repository_type'].unique()
            
            accuracy_by_type = []
            type_labels = []
            
            for repo_type in repo_types:
                type_data = df[df['repository_type'] == repo_type]
                fastpath_data = type_data[type_data['system_name'].str.contains('fastpath', case=False)]
                baseline_data = type_data[~type_data['system_name'].str.contains('fastpath', case=False)]
                
                if len(fastpath_data) > 0 and len(baseline_data) > 0:
                    fastpath_acc = fastpath_data['qa_accuracy'].mean()
                    baseline_acc = baseline_data['qa_accuracy'].mean()
                    
                    ax2.scatter([repo_type], [fastpath_acc], c='blue', s=100, alpha=0.7, label='FastPath' if repo_type == repo_types[0] else "")
                    ax2.scatter([repo_type], [baseline_acc], c='red', s=100, alpha=0.7, label='Baselines' if repo_type == repo_types[0] else "")
            
            ax2.set_xlabel('Repository Type')
            ax2.set_ylabel('Mean QA Accuracy')
            ax2.set_title('Accuracy by Repository Type')
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        plot_file = self.output_dir / 'figures' / 'qa_accuracy_analysis.pdf'
        plt.savefig(plot_file, format='pdf')
        plt.close()
        
        return str(plot_file)
    
    def generate_memory_usage_analysis(self, df: pd.DataFrame) -> str:
        """Generate memory usage analysis."""
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Memory usage vs QA accuracy scatter plot
        systems = df['system_name'].unique()
        
        for system in systems:
            system_data = df[df['system_name'] == system]
            if len(system_data) > 0:
                x = system_data['memory_usage_bytes'] / (1024 * 1024)  # Convert to MB
                y = system_data['qa_accuracy']
                
                ax.scatter(x, y, 
                          c=PUBLICATION_COLORS.get(system, '#cccccc'),
                          label=self._format_system_name(system),
                          alpha=0.7, s=60)
        
        ax.set_xlabel('Memory Usage (MB)')
        ax.set_ylabel('QA Accuracy')
        ax.set_title('Memory Usage vs QA Accuracy Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        plt.tight_layout()
        
        plot_file = self.output_dir / 'figures' / 'memory_usage_analysis.pdf'
        plt.savefig(plot_file, format='pdf')
        plt.close()
        
        return str(plot_file)
    
    def generate_bootstrap_intervals_plot(self, bootstrap_intervals: Dict[str, Dict[str, Any]]) -> str:
        """Generate bootstrap confidence intervals plot."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        metrics = ['qa_accuracy', 'qa_f1_score', 'execution_time_seconds', 'memory_usage_bytes']
        metric_titles = ['QA Accuracy', 'F1 Score', 'Execution Time (s)', 'Memory Usage (MB)']
        
        for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[idx]
            
            systems = []
            statistics = []
            ci_lowers = []
            ci_uppers = []
            
            for system, intervals in bootstrap_intervals.items():
                if metric in intervals:
                    interval_data = intervals[metric]
                    
                    statistic = interval_data['statistic']
                    ci_lower, ci_upper = interval_data['confidence_interval']
                    
                    # Convert bytes to MB for memory
                    if metric == 'memory_usage_bytes':
                        statistic /= (1024 * 1024)
                        ci_lower /= (1024 * 1024)
                        ci_upper /= (1024 * 1024)
                    
                    systems.append(self._format_system_name(system))
                    statistics.append(statistic)
                    ci_lowers.append(ci_lower)
                    ci_uppers.append(ci_upper)
            
            if systems:
                y_pos = range(len(systems))
                
                # Create error bars
                errors = [[stat - ci_low for stat, ci_low in zip(statistics, ci_lowers)],
                         [ci_high - stat for stat, ci_high in zip(statistics, ci_uppers)]]
                
                ax.barh(y_pos, statistics, xerr=errors, capsize=5,
                       color=[PUBLICATION_COLORS.get(s.lower().replace(' ', '_').replace('-', '_'), '#cccccc') for s in systems],
                       alpha=0.7, edgecolor='black', linewidth=0.5)
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(systems)
                ax.set_xlabel(title)
                ax.set_title(f'{title} - Bootstrap 95% CI')
                ax.grid(True, alpha=0.3, axis='x')
                
                if metric == 'execution_time_seconds':
                    ax.set_xscale('log')
        
        plt.tight_layout()
        
        plot_file = self.output_dir / 'figures' / 'bootstrap_intervals.pdf'
        plt.savefig(plot_file, format='pdf')
        plt.close()
        
        return str(plot_file)
    
    def generate_system_comparison_heatmap(self, system_summaries: Dict[str, Dict[str, Any]]) -> str:
        """Generate system comparison heatmap."""
        
        # Prepare data for heatmap
        systems = list(system_summaries.keys())
        metrics = ['qa_accuracy', 'qa_f1_score', 'execution_time_seconds', 'memory_usage_bytes']
        metric_labels = ['QA Accuracy', 'F1 Score', 'Exec Time', 'Memory']
        
        # Create data matrix (normalized to 0-1 scale)
        data_matrix = np.zeros((len(systems), len(metrics)))
        
        for i, system in enumerate(systems):
            for j, metric in enumerate(metrics):
                if metric in system_summaries[system]:
                    value = system_summaries[system][metric]['mean']
                    data_matrix[i, j] = value
        
        # Normalize each metric column
        for j in range(len(metrics)):
            col = data_matrix[:, j]
            if col.max() > col.min():
                # For time and memory, lower is better (invert)
                if metrics[j] in ['execution_time_seconds', 'memory_usage_bytes']:
                    col = 1 / (1 + col / col.max())  # Transform to "higher is better"
                
                # Normalize to 0-1
                col = (col - col.min()) / (col.max() - col.min())
                data_matrix[:, j] = col
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metric_labels)
        ax.set_yticks(range(len(systems)))
        ax.set_yticklabels([self._format_system_name(s) for s in systems])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Performance (0=Worst, 1=Best)')
        
        # Add value annotations
        for i in range(len(systems)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('System Performance Heatmap')
        plt.tight_layout()
        
        plot_file = self.output_dir / 'figures' / 'system_comparison_heatmap.pdf'
        plt.savefig(plot_file, format='pdf')
        plt.close()
        
        return str(plot_file)
    
    def export_research_data(
        self, 
        raw_measurements: List[Dict[str, Any]], 
        statistical_results: Dict[str, Any]
    ) -> List[str]:
        """Export research data in multiple formats for transparency."""
        
        exported_files = []
        
        # 1. Raw measurements CSV
        df = pd.DataFrame(raw_measurements)
        csv_file = self.output_dir / 'data' / 'raw_measurements.csv'
        df.to_csv(csv_file, index=False)
        exported_files.append(str(csv_file))
        
        # 2. Statistical results JSON
        json_file = self.output_dir / 'data' / 'statistical_results.json'
        with open(json_file, 'w') as f:
            json.dump(statistical_results, f, indent=2, default=str)
        exported_files.append(str(json_file))
        
        # 3. Summary statistics CSV
        if 'system_summaries' in statistical_results:
            summary_data = []
            for system, metrics in statistical_results['system_summaries'].items():
                for metric, stats in metrics.items():
                    summary_data.append({
                        'system': system,
                        'metric': metric,
                        'mean': stats['mean'],
                        'std': stats['std'],
                        'median': stats['median'],
                        'count': stats['count'],
                        'min': stats['min'],
                        'max': stats['max']
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_file = self.output_dir / 'data' / 'summary_statistics.csv'
            summary_df.to_csv(summary_file, index=False)
            exported_files.append(str(summary_file))
        
        # 4. Effect sizes CSV
        if 'effect_sizes' in statistical_results:
            effect_data = []
            for fastpath_system, comparisons in statistical_results['effect_sizes'].items():
                if isinstance(comparisons, dict):
                    for comparison, metrics in comparisons.items():
                        baseline = comparison.replace('vs_', '')
                        for metric, effect_info in metrics.items():
                            if isinstance(effect_info, dict):
                                effect_data.append({
                                    'fastpath_system': fastpath_system,
                                    'baseline_system': baseline,
                                    'metric': metric,
                                    'cohens_d': effect_info.get('value', 0),
                                    'magnitude': effect_info.get('magnitude', 'unknown'),
                                    'ci_lower': effect_info.get('confidence_interval', [0, 0])[0],
                                    'ci_upper': effect_info.get('confidence_interval', [0, 0])[1]
                                })
            
            if effect_data:
                effect_df = pd.DataFrame(effect_data)
                effect_file = self.output_dir / 'data' / 'effect_sizes.csv'
                effect_df.to_csv(effect_file, index=False)
                exported_files.append(str(effect_file))
        
        return exported_files
    
    def generate_latex_document_template(
        self, 
        generated_files: Dict[str, List[str]], 
        config: Any
    ) -> str:
        """Generate LaTeX document template with all figures and tables."""
        
        latex_content = []
        
        # Document header
        latex_content.extend([
            "\\documentclass[conference]{IEEEtran}",
            "\\usepackage{graphicx}",
            "\\usepackage{booktabs}",
            "\\usepackage{amsmath}",
            "\\usepackage{caption}",
            "\\usepackage{subcaption}",
            "",
            "\\title{FastPath: Intelligent Code Context Selection for Large Language Models}",
            "\\author{",
            "\\IEEEauthorblockN{Research Team}",
            "\\IEEEauthorblockA{Institution Name\\\\",
            "Email: research@institution.edu}",
            "}",
            "",
            "\\begin{document}",
            "\\maketitle",
            "",
            "\\begin{abstract}",
            "This paper presents FastPath, an intelligent system for selecting relevant code context",
            "for large language model (LLM) queries within token budget constraints. Our evaluation",
            f"across {getattr(config, 'min_repositories_per_type', 5)} repositories per type demonstrates",
            "significant improvements in QA accuracy while maintaining computational efficiency.",
            "\\end{abstract}",
            "",
            "\\section{Introduction}",
            "% Introduction content here",
            "",
            "\\section{Methodology}",
            "% Methodology content here",
            "",
            "\\section{Results}",
            ""
        ])
        
        # Include performance table
        latex_content.extend([
            "\\subsection{Performance Comparison}",
            "Table~\\ref{tab:performance_comparison} shows the performance comparison across all systems.",
            "",
            "\\input{tables/performance_table.tex}",
            ""
        ])
        
        # Include significance tests
        latex_content.extend([
            "\\subsection{Statistical Significance}",
            "Table~\\ref{tab:significance_tests} presents the statistical significance tests.",
            "",
            "\\input{tables/significance_table.tex}",
            ""
        ])
        
        # Include effect sizes
        latex_content.extend([
            "\\subsection{Effect Sizes}",
            "Table~\\ref{tab:effect_sizes} shows the effect sizes comparing FastPath to baselines.",
            "",
            "\\input{tables/effect_sizes_table.tex}",
            ""
        ])
        
        # Include figures
        latex_content.extend([
            "\\subsection{Performance Analysis}",
            "Figure~\\ref{fig:performance_comparison} illustrates the performance comparison across metrics.",
            "",
            "\\begin{figure*}[t]",
            "\\centering",
            "\\includegraphics[width=\\textwidth]{figures/performance_comparison.pdf}",
            "\\caption{Performance comparison across all metrics.}",
            "\\label{fig:performance_comparison}",
            "\\end{figure*}",
            "",
            "\\begin{figure}[t]",
            "\\centering",
            "\\includegraphics[width=\\columnwidth]{figures/qa_accuracy_analysis.pdf}",
            "\\caption{QA accuracy analysis with confidence intervals.}",
            "\\label{fig:qa_accuracy}",
            "\\end{figure}",
            "",
            "\\begin{figure}[t]",
            "\\centering",
            "\\includegraphics[width=\\columnwidth]{figures/execution_time_analysis.pdf}",
            "\\caption{Execution time analysis and speed improvements.}",
            "\\label{fig:execution_time}",
            "\\end{figure}",
            ""
        ])
        
        # Document footer
        latex_content.extend([
            "\\section{Conclusion}",
            "% Conclusion content here",
            "",
            "\\section{Acknowledgments}",
            "% Acknowledgments here",
            "",
            "\\bibliographystyle{IEEEtran}",
            "\\bibliography{references}",
            "",
            "\\end{document}"
        ])
        
        # Save LaTeX file
        latex_file = self.output_dir / 'latex' / 'fastpath_paper.tex'
        with open(latex_file, 'w') as f:
            f.write('\n'.join(latex_content))
        
        return str(latex_file)
    
    def generate_results_summary(self, statistical_results: Dict[str, Any], config: Any) -> str:
        """Generate executive summary of results."""
        
        summary_content = []
        
        # Header
        summary_content.extend([
            "# FastPath Evaluation Results Summary",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Configuration: {getattr(config, 'name', 'Unknown')}",
            "",
            "## Key Findings",
            ""
        ])
        
        # Extract key findings from statistical results
        if 'summary' in statistical_results and 'key_findings' in statistical_results['summary']:
            for finding in statistical_results['summary']['key_findings']:
                metric = finding['metric']
                best_system = finding['best_system']
                best_value = finding['best_value']
                improvement_type = finding['improvement_type']
                
                summary_content.append(f"- **{metric}**: {best_system} achieved {best_value:.4f} ({improvement_type})")
        
        summary_content.append("")
        
        # Statistical significance summary
        if 'significance_tests' in statistical_results:
            significant_tests = sum(1 for test in statistical_results['significance_tests'].values() 
                                  if isinstance(test, dict) and test.get('is_significant', False))
            total_tests = len(statistical_results['significance_tests'])
            
            summary_content.extend([
                "## Statistical Significance",
                f"- {significant_tests}/{total_tests} tests showed statistical significance (p < 0.05)",
                ""
            ])
        
        # Effect sizes summary
        if 'effect_sizes' in statistical_results:
            large_effects = 0
            medium_effects = 0
            
            for fastpath_system, comparisons in statistical_results['effect_sizes'].items():
                if isinstance(comparisons, dict):
                    for comparison, metrics in comparisons.items():
                        for metric, effect_data in metrics.items():
                            if isinstance(effect_data, dict):
                                magnitude = effect_data.get('magnitude', '')
                                if magnitude == 'large':
                                    large_effects += 1
                                elif magnitude == 'medium':
                                    medium_effects += 1
            
            summary_content.extend([
                "## Effect Sizes",
                f"- {large_effects} comparisons showed large effect sizes (Cohen's d > 0.8)",
                f"- {medium_effects} comparisons showed medium effect sizes (Cohen's d > 0.5)",
                ""
            ])
        
        # Recommendations
        if 'summary' in statistical_results and 'recommendations' in statistical_results['summary']:
            summary_content.extend([
                "## Recommendations",
                ""
            ])
            for recommendation in statistical_results['summary']['recommendations']:
                summary_content.append(f"- {recommendation}")
        
        # Save summary
        summary_file = self.output_dir / 'results_summary.md'
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_content))
        
        return str(summary_file)
    
    def _create_boxplot(self, df: pd.DataFrame, metric: str, ylabel: str, ax, log_scale: bool = False):
        """Create a box plot for a metric."""
        systems = sorted(df['system_name'].unique())
        data = []
        labels = []
        
        for system in systems:
            system_data = df[df['system_name'] == system][metric].dropna()
            if len(system_data) > 0:
                data.append(system_data)
                labels.append(self._format_system_name(system))
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            
            # Color boxes
            for patch, system in zip(bp['boxes'], systems):
                patch.set_facecolor(PUBLICATION_COLORS.get(system, '#cccccc'))
                patch.set_alpha(0.7)
            
            ax.set_ylabel(ylabel)
            ax.tick_params(axis='x', rotation=45)
            
            if log_scale:
                ax.set_yscale('log')
            
            ax.grid(True, alpha=0.3)
    
    def _format_system_name(self, system_name: str) -> str:
        """Format system name for display."""
        name_map = {
            'fastpath_v1': 'FastPath V1',
            'fastpath_v2': 'FastPath V2', 
            'fastpath_v3': 'FastPath V3',
            'naive_tfidf': 'Naive TF-IDF',
            'bm25': 'BM25',
            'random': 'Random'
        }
        return name_map.get(system_name, system_name.replace('_', ' ').title())
    
    def _is_best_performance(
        self, 
        system: str, 
        metric: str, 
        system_summaries: Dict[str, Dict[str, Any]], 
        higher_better: bool
    ) -> bool:
        """Check if system has best performance for metric."""
        values = []
        for sys, summary in system_summaries.items():
            if metric in summary:
                values.append(summary[metric]['mean'])
        
        if not values:
            return False
        
        system_value = system_summaries[system][metric]['mean']
        
        if higher_better:
            return system_value == max(values)
        else:
            return system_value == min(values)


# Export main class
__all__ = ['PublicationDataGenerator']
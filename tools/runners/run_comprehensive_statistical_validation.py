#!/usr/bin/env python3
"""
Comprehensive Statistical Validation Runner for FastPath Research
================================================================

This script executes the complete peer-review quality statistical analysis
for FastPath research validation. It demonstrates all advanced statistical
methods suitable for publication in top-tier academic venues.

Features:
- Comprehensive experimental design validation
- Multiple baseline comparison systems  
- Cross-validation with repository stratification
- Advanced statistical testing with assumption validation
- Multiple comparison correction using state-of-the-art methods
- Bayesian analysis and probability of superiority
- Forest plots and publication-ready visualizations
- Complete reproducibility package generation

Usage:
    python run_comprehensive_statistical_validation.py [options]
    
Options:
    --demo              Run with demonstration data
    --baseline-file     Path to baseline performance data (JSON)
    --fastpath-file     Path to FastPath performance data (JSON)
    --output-dir        Output directory for results and artifacts
    --significance      Significance level (default: 0.05)
    --target-improvement Target improvement threshold (default: 0.20)
    --bootstrap-n       Bootstrap iterations (default: 10000)
    --power-threshold   Statistical power threshold (default: 0.8)
    
Examples:
    # Run demonstration with synthetic data
    python run_comprehensive_statistical_validation.py --demo
    
    # Run with real data files
    python run_comprehensive_statistical_validation.py \
        --baseline-file baseline_performance.json \
        --fastpath-file fastpath_performance.json \
        --output-dir statistical_results/
        
    # High-precision analysis with custom parameters
    python run_comprehensive_statistical_validation.py --demo \
        --significance 0.01 \
        --bootstrap-n 50000 \
        --target-improvement 0.25
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd
import warnings

# Import our comprehensive statistical framework
from peer_review_statistical_framework import (
    PeerReviewStatisticalFramework,
    ExperimentalDesign,
    logger
)

warnings.filterwarnings('ignore')

def create_experimental_design(args: argparse.Namespace) -> ExperimentalDesign:
    """Create formal experimental design specification."""
    
    return ExperimentalDesign(
        study_name="FastPath Performance Enhancement Validation",
        research_questions=[
            "How much does FastPath improve QA accuracy vs established baselines?",
            "Which specific FastPath enhancements contribute most to performance gains?", 
            "How does performance vary across different repository characteristics?",
            "What are the computational trade-offs between speed and accuracy?",
            "How robust are improvements across different question types?"
        ],
        primary_hypothesis="FastPath variants achieve ≥20% improvement in QA accuracy compared to BM25 baseline (p<0.05)",
        secondary_hypotheses=[
            "FastPath V3 outperforms FastPath V2 significantly", 
            "Performance gains are consistent across repository types",
            "Effect sizes are large (Cohen's d ≥ 0.8) for primary comparisons"
        ],
        baseline_systems=["BM25", "TF-IDF", "Random", "Naive"],
        treatment_systems=["FastPath_V2", "FastPath_V3"],
        repository_types=["CLI_Tools", "Libraries", "Web_Apps", "Data_Science"],
        sample_sizes={},  # Will be populated from actual data
        alpha_level=args.significance,
        power_target=args.power_threshold,
        effect_size_target=0.5,  # Medium effect size
        multiple_comparison_method="benjamini_hochberg",
        cross_validation_folds=10,
        bootstrap_iterations=args.bootstrap_n,
        permutation_test_iterations=10000,
        random_seeds=list(range(42, 52)),  # 10 seeds for robust validation
        environment_specs={
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
            'numpy_version': np.__version__,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
    )

def load_performance_data(file_path: Path) -> Dict[str, np.ndarray]:
    """Load performance data from JSON file."""
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert to numpy arrays
        performance_data = {}
        for system_name, values in data.items():
            if isinstance(values, list) and len(values) > 0:
                performance_data[system_name] = np.array(values, dtype=float)
            elif isinstance(values, dict) and 'qa_accuracy_per_100k' in values:
                # Handle nested format
                performance_data[system_name] = np.array(values['qa_accuracy_per_100k'], dtype=float)
            else:
                logger.warning(f"Skipping invalid data for system: {system_name}")
        
        logger.info(f"Loaded performance data for {len(performance_data)} systems from {file_path}")
        return performance_data
        
    except Exception as e:
        logger.error(f"Error loading performance data from {file_path}: {e}")
        raise

def generate_synthetic_data(design: ExperimentalDesign, 
                          n_observations: int = 50) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Generate realistic synthetic performance data for demonstration."""
    
    logger.info(f"Generating synthetic performance data with n={n_observations}")
    
    # Set seed for reproducibility
    np.random.seed(design.random_seeds[0])
    
    # Baseline systems with realistic performance distributions
    baseline_systems = {
        'bm25': np.random.normal(0.652, 0.078, n_observations),      # Strong baseline
        'tfidf': np.random.normal(0.618, 0.089, n_observations),     # Moderate baseline
        'naive_tfidf': np.random.normal(0.545, 0.095, n_observations), # Weak baseline
        'random': np.random.normal(0.412, 0.125, n_observations)     # Random baseline
    }
    
    # FastPath systems with targeted improvements
    fastpath_systems = {
        'fastpath_v2': np.random.normal(0.785, 0.065, n_observations),  # 20.4% over BM25
        'fastpath_v3': np.random.normal(0.825, 0.058, n_observations)   # 26.5% over BM25
    }
    
    # Ensure no negative values (performance can't be negative)
    for systems in [baseline_systems, fastpath_systems]:
        for name, data in systems.items():
            systems[name] = np.clip(data, 0.0, 1.0)  # Clip to valid range
    
    # Log data characteristics
    logger.info("Generated synthetic data characteristics:")
    logger.info("Baseline Systems:")
    for name, data in baseline_systems.items():
        logger.info(f"  {name}: mean={np.mean(data):.3f}, std={np.std(data):.3f}")
    
    logger.info("FastPath Systems:")
    for name, data in fastpath_systems.items():
        improvement_vs_bm25 = ((np.mean(data) - np.mean(baseline_systems['bm25'])) / 
                              np.mean(baseline_systems['bm25'])) * 100
        logger.info(f"  {name}: mean={np.mean(data):.3f}, std={np.std(data):.3f}, "
                   f"improvement={improvement_vs_bm25:.1f}%")
    
    return baseline_systems, fastpath_systems

def generate_repository_metadata(n_repos: int = 50) -> Dict[str, Any]:
    """Generate realistic repository metadata for heterogeneity analysis."""
    
    np.random.seed(42)
    
    # Repository types with realistic distributions
    repo_types = np.random.choice(
        ['cli_tool', 'library', 'web_app', 'data_science', 'mobile_app'], 
        size=n_repos, 
        p=[0.3, 0.25, 0.2, 0.15, 0.1]
    )
    
    # Programming languages
    languages = np.random.choice(
        ['python', 'javascript', 'java', 'typescript', 'go', 'rust'], 
        size=n_repos,
        p=[0.35, 0.25, 0.15, 0.1, 0.08, 0.07]
    )
    
    # Repository sizes (lines of code)
    repo_sizes = np.random.lognormal(mean=8.5, sigma=1.2, size=n_repos).astype(int)
    repo_sizes = np.clip(repo_sizes, 1000, 500000)  # Reasonable bounds
    
    # Complexity metrics
    complexity_scores = np.random.normal(loc=6.5, scale=2.1, size=n_repos)
    complexity_scores = np.clip(complexity_scores, 1.0, 10.0)
    
    return {
        'repository_count': n_repos,
        'types': repo_types.tolist(),
        'languages': languages.tolist(), 
        'sizes_loc': repo_sizes.tolist(),
        'complexity_scores': complexity_scores.tolist(),
        'type_distribution': {
            repo_type: int(np.sum(repo_types == repo_type)) 
            for repo_type in np.unique(repo_types)
        },
        'language_distribution': {
            lang: int(np.sum(languages == lang))
            for lang in np.unique(languages)
        }
    }

def run_cross_validation_analysis(framework: PeerReviewStatisticalFramework,
                                 baseline_systems: Dict[str, np.ndarray],
                                 fastpath_systems: Dict[str, np.ndarray],
                                 n_folds: int = 10) -> Dict[str, Any]:
    """Run cross-validation analysis to assess robustness."""
    
    logger.info(f"Running {n_folds}-fold cross-validation analysis")
    
    cv_results = {
        'n_folds': n_folds,
        'fold_results': [],
        'stability_metrics': {}
    }
    
    # Combine all systems for fold creation
    all_systems = {**baseline_systems, **fastpath_systems}
    min_size = min(len(data) for data in all_systems.values())
    
    if min_size < n_folds:
        logger.warning(f"Sample size ({min_size}) too small for {n_folds}-fold CV, reducing to {min_size}")
        n_folds = max(2, min_size // 2)
    
    fold_size = min_size // n_folds
    
    fold_improvements = []
    fold_effect_sizes = []
    fold_p_values = []
    
    for fold in range(n_folds):
        logger.info(f"Processing fold {fold + 1}/{n_folds}")
        
        # Create fold indices
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else min_size
        fold_indices = list(range(start_idx, end_idx))
        
        # Extract fold data
        fold_baselines = {
            name: data[fold_indices] 
            for name, data in baseline_systems.items()
        }
        fold_fastpath = {
            name: data[fold_indices]
            for name, data in fastpath_systems.items()
        }
        
        # Run analysis on fold
        try:
            fold_results = framework.comprehensive_fastpath_analysis(
                baseline_systems=fold_baselines,
                fastpath_systems=fold_fastpath,
                target_improvement=0.20
            )
            
            cv_results['fold_results'].append({
                'fold': fold,
                'sample_size': len(fold_indices),
                'results': fold_results
            })
            
            # Extract key metrics for stability assessment
            primary_comps = fold_results.get('primary_comparisons', {})
            if 'fastpath_v2_vs_bm25' in primary_comps:
                comp = primary_comps['fastpath_v2_vs_bm25']
                fold_improvements.append(comp.get('improvement_percentage', 0))
                fold_effect_sizes.append(comp['effect_sizes']['cohens_d']['cohens_d'])
                
                # Get p-value
                if comp['parametric_test']['p_value'] is not None:
                    fold_p_values.append(comp['parametric_test']['p_value'])
                else:
                    fold_p_values.append(comp['non_parametric_test']['p_value'])
            
        except Exception as e:
            logger.error(f"Error processing fold {fold}: {e}")
            cv_results['fold_results'].append({
                'fold': fold,
                'sample_size': len(fold_indices),
                'error': str(e)
            })
    
    # Calculate stability metrics
    if fold_improvements:
        cv_results['stability_metrics'] = {
            'improvement_percentage': {
                'mean': float(np.mean(fold_improvements)),
                'std': float(np.std(fold_improvements)),
                'coefficient_of_variation': float(np.std(fold_improvements) / np.mean(fold_improvements)),
                'range': [float(np.min(fold_improvements)), float(np.max(fold_improvements))]
            },
            'effect_sizes': {
                'mean': float(np.mean(fold_effect_sizes)),
                'std': float(np.std(fold_effect_sizes)),
                'coefficient_of_variation': float(np.std(fold_effect_sizes) / np.mean(fold_effect_sizes)) if np.mean(fold_effect_sizes) != 0 else np.inf,
                'range': [float(np.min(fold_effect_sizes)), float(np.max(fold_effect_sizes))]
            },
            'p_values': {
                'mean': float(np.mean(fold_p_values)),
                'proportion_significant': float(np.mean(np.array(fold_p_values) < 0.05)),
                'range': [float(np.min(fold_p_values)), float(np.max(fold_p_values))]
            },
            'consistency_assessment': 'high' if np.std(fold_improvements) / np.mean(fold_improvements) < 0.2 else 'moderate' if np.std(fold_improvements) / np.mean(fold_improvements) < 0.5 else 'low'
        }
    
    logger.info("Cross-validation analysis completed")
    return cv_results

def generate_comprehensive_report(analysis_results: Dict[str, Any],
                                cv_results: Dict[str, Any],
                                experimental_design: ExperimentalDesign,
                                output_dir: Path) -> Path:
    """Generate comprehensive analysis report."""
    
    report_path = output_dir / "comprehensive_statistical_report.md"
    
    # Extract key results
    pub_assessment = analysis_results.get('publication_assessment', {})
    primary_comps = analysis_results.get('primary_comparisons', {})
    mc_correction = analysis_results.get('multiple_comparison_correction', {})
    power_analysis = analysis_results.get('power_analysis', {})
    
    report_content = f"""# Comprehensive Statistical Analysis Report
## FastPath Performance Enhancement Validation

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Framework Version:** Peer-Review Statistical Framework v1.0

---

## Executive Summary

### Publication Readiness Assessment
- **Overall Score:** {pub_assessment.get('overall_score', 0):.1f}/100
- **Primary Hypothesis:** {'✅ SUPPORTED' if pub_assessment.get('primary_hypothesis_supported', False) else '❌ NOT SUPPORTED'}
- **Statistical Rigor:** {pub_assessment.get('statistical_rigor_score', 0):.1f}/100
- **Power Adequacy:** {'✅ ADEQUATE' if pub_assessment.get('sample_size_adequate', False) else '❌ INADEQUATE'}
- **Effect Size:** {'✅ ADEQUATE' if pub_assessment.get('effect_size_adequate', False) else '❌ INADEQUATE'}

### Recommended Publication Venues
{chr(10).join(f"- {venue}" for venue in pub_assessment.get('suitable_venues', [])[:5])}

**Estimated Acceptance Probability:** {pub_assessment.get('estimated_acceptance_probability', 0):.1%}

---

## Research Design

### Primary Research Question
{experimental_design.primary_hypothesis}

### Secondary Research Questions
{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(experimental_design.research_questions))}

### Experimental Parameters
- **Significance Level (α):** {experimental_design.alpha_level}
- **Power Threshold:** {experimental_design.power_target}
- **Target Effect Size:** {experimental_design.effect_size_target}
- **Bootstrap Iterations:** {experimental_design.bootstrap_iterations:,}
- **Multiple Comparison Method:** {experimental_design.multiple_comparison_method}

---

## Statistical Results

### Primary Comparisons Summary
"""
    
    # Add primary comparison results
    if primary_comps:
        report_content += "\n| Comparison | Improvement (%) | Cohen's d | 95% CI | p-value | Significant |\n"
        report_content += "|------------|----------------|-----------|--------|---------|-------------|\n"
        
        for comp_name, comp_data in primary_comps.items():
            improvement = comp_data.get('improvement_percentage', 0)
            cohens_d = comp_data['effect_sizes']['cohens_d']['cohens_d']
            ci_lower, ci_upper = comp_data['effect_sizes']['cohens_d']['confidence_interval']
            
            # Get p-value
            if comp_data['parametric_test']['p_value'] is not None:
                p_value = comp_data['parametric_test']['p_value']
            else:
                p_value = comp_data['non_parametric_test']['p_value']
            
            significant = "✅" if p_value < 0.05 else "❌"
            p_str = f"{p_value:.4f}" if p_value >= 0.001 else "< 0.001"
            
            clean_name = comp_name.replace('_', ' ').title()
            report_content += f"| {clean_name} | {improvement:.1f} | {cohens_d:.2f} | [{ci_lower:.2f}, {ci_upper:.2f}] | {p_str} | {significant} |\n"
    
    # Add multiple comparison correction results
    if mc_correction:
        report_content += f"""
### Multiple Comparison Correction
- **Method:** {mc_correction.get('method', 'N/A').replace('_', ' ').title()}
- **Total Comparisons:** {mc_correction.get('total_comparisons', 0)}
- **Significant After Correction:** {mc_correction.get('significant_comparisons', 0)}/{mc_correction.get('total_comparisons', 0)}
- **Family-Wise Error Rate:** {mc_correction.get('family_wise_error_rate', 0):.4f}
"""
    
    # Add power analysis results
    if power_analysis:
        report_content += f"""
### Power Analysis Summary
- **Mean Observed Power:** {power_analysis.get('mean_power', 0):.3f}
- **Power Range:** {power_analysis.get('min_power', 0):.3f} - {power_analysis.get('max_power', 0):.3f}
- **Adequate Power Proportion:** {power_analysis.get('adequate_power_proportion', 0):.1%}
- **Overall Power Assessment:** {'✅ ADEQUATE' if power_analysis.get('overall_power_adequate', False) else '❌ NEEDS IMPROVEMENT'}
"""
    
    # Add cross-validation results
    if cv_results and 'stability_metrics' in cv_results:
        stability = cv_results['stability_metrics']
        improvement_stats = stability.get('improvement_percentage', {})
        effect_stats = stability.get('effect_sizes', {})
        p_value_stats = stability.get('p_values', {})
        
        report_content += f"""
### Cross-Validation Stability Analysis
- **Number of Folds:** {cv_results.get('n_folds', 'N/A')}
- **Consistency Assessment:** {stability.get('consistency_assessment', 'N/A').upper()}

#### Improvement Percentage Stability
- **Mean:** {improvement_stats.get('mean', 0):.2f}%
- **Standard Deviation:** {improvement_stats.get('std', 0):.2f}%
- **Coefficient of Variation:** {improvement_stats.get('coefficient_of_variation', 0):.3f}

#### Effect Size Stability  
- **Mean Cohen's d:** {effect_stats.get('mean', 0):.3f}
- **Standard Deviation:** {effect_stats.get('std', 0):.3f}
- **Coefficient of Variation:** {effect_stats.get('coefficient_of_variation', 0):.3f}

#### Statistical Significance Stability
- **Proportion Significant:** {p_value_stats.get('proportion_significant', 0):.1%}
- **Mean p-value:** {p_value_stats.get('mean', 0):.4f}
"""
    
    # Add methodology assessment
    report_content += f"""
---

## Methodology Assessment

### Statistical Rigor Components
"""
    
    if pub_assessment.get('statistical_rigor', {}):
        rigor_components = pub_assessment['statistical_rigor']
        for component, status in rigor_components.items():
            status_icon = "✅" if status else "❌"
            clean_component = component.replace('_', ' ').title()
            report_content += f"- **{clean_component}:** {status_icon}\n"
    
    # Add strengths and limitations
    strengths = pub_assessment.get('methodology_strengths', [])
    limitations = pub_assessment.get('potential_limitations', [])
    concerns = pub_assessment.get('reviewer_concerns', [])
    recommendations = pub_assessment.get('recommendations', [])
    
    if strengths:
        report_content += f"""
### Methodology Strengths
{chr(10).join(f"- {strength}" for strength in strengths)}
"""
    
    if limitations:
        report_content += f"""
### Potential Limitations
{chr(10).join(f"- {limitation}" for limitation in limitations)}
"""
    
    if concerns:
        report_content += f"""
### Potential Reviewer Concerns
{chr(10).join(f"- {concern}" for concern in concerns)}
"""
    
    if recommendations:
        report_content += f"""
### Recommendations for Improvement
{chr(10).join(f"- {recommendation}" for recommendation in recommendations)}
"""
    
    # Add conclusion
    conclusion = "READY FOR SUBMISSION" if pub_assessment.get('overall_score', 0) >= 80 else "REQUIRES IMPROVEMENT"
    report_content += f"""
---

## Conclusion

**Publication Status:** {conclusion}

This comprehensive statistical analysis {'meets' if conclusion == 'READY FOR SUBMISSION' else 'does not fully meet'} the standards expected for peer-reviewed publication in top-tier academic venues. {'The evidence strongly supports the primary research hypothesis with appropriate statistical rigor.' if conclusion == 'READY FOR SUBMISSION' else 'Additional work is recommended to strengthen the statistical evidence before submission.'}

### Next Steps
{'1. Submit to target venue with confidence' if conclusion == 'READY FOR SUBMISSION' else '1. Address identified limitations and concerns'}
{'2. Prepare response to potential reviewer comments' if conclusion == 'READY FOR SUBMISSION' else '2. Consider additional data collection or analysis'}
{'3. Share reproducibility materials with reviewers' if conclusion == 'READY FOR SUBMISSION' else '3. Strengthen methodology based on recommendations'}

---

*Report generated by Peer-Review Statistical Framework v1.0*
*Analysis timestamp: {analysis_results.get('timestamp', 'N/A')}*
"""
    
    # Write report
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Comprehensive report generated: {report_path}")
    return report_path

def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description="Comprehensive Statistical Validation for FastPath Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run demonstration with synthetic data
    python run_comprehensive_statistical_validation.py --demo
    
    # Run with custom parameters
    python run_comprehensive_statistical_validation.py --demo \\
        --significance 0.01 --target-improvement 0.25 --bootstrap-n 50000
    
    # Run with real data files
    python run_comprehensive_statistical_validation.py \\
        --baseline-file baseline_data.json \\
        --fastpath-file fastpath_data.json \\
        --output-dir results/
        """
    )
    
    parser.add_argument('--demo', action='store_true', 
                       help='Run with demonstration synthetic data')
    parser.add_argument('--baseline-file', type=Path,
                       help='Path to baseline performance data JSON file')
    parser.add_argument('--fastpath-file', type=Path,
                       help='Path to FastPath performance data JSON file')
    parser.add_argument('--output-dir', type=Path, default=Path('comprehensive_statistical_results'),
                       help='Output directory for results and artifacts')
    parser.add_argument('--significance', type=float, default=0.05,
                       help='Significance level (alpha) for statistical tests')
    parser.add_argument('--target-improvement', type=float, default=0.20,
                       help='Target improvement threshold (e.g., 0.20 for 20%%)')
    parser.add_argument('--bootstrap-n', type=int, default=10000,
                       help='Number of bootstrap iterations')
    parser.add_argument('--power-threshold', type=float, default=0.8,
                       help='Statistical power threshold')
    parser.add_argument('--cross-validation-folds', type=int, default=10,
                       help='Number of cross-validation folds')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE STATISTICAL VALIDATION FOR FASTPATH RESEARCH")
    logger.info("="*80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Significance level: {args.significance}")
    logger.info(f"Target improvement: {args.target_improvement:.1%}")
    logger.info(f"Bootstrap iterations: {args.bootstrap_n:,}")
    
    try:
        # Create experimental design
        experimental_design = create_experimental_design(args)
        logger.info("✅ Experimental design created")
        
        # Load or generate data
        if args.demo:
            logger.info("🔬 Generating synthetic demonstration data...")
            baseline_systems, fastpath_systems = generate_synthetic_data(experimental_design)
            repository_metadata = generate_repository_metadata()
        else:
            if not args.baseline_file or not args.fastpath_file:
                raise ValueError("--baseline-file and --fastpath-file are required when not using --demo")
            
            logger.info(f"📁 Loading baseline data from: {args.baseline_file}")
            baseline_systems = load_performance_data(args.baseline_file)
            
            logger.info(f"📁 Loading FastPath data from: {args.fastpath_file}")
            fastpath_systems = load_performance_data(args.fastpath_file)
            
            repository_metadata = None  # Would be loaded from additional file if available
        
        # Initialize statistical framework
        logger.info("🔧 Initializing statistical framework...")
        framework = PeerReviewStatisticalFramework(
            alpha=args.significance,
            power_threshold=args.power_threshold,
            effect_size_threshold=0.5,  # Medium effect size
            bootstrap_iterations=args.bootstrap_n
        )
        
        # Run comprehensive analysis
        logger.info("📊 Executing comprehensive statistical analysis...")
        analysis_results = framework.comprehensive_fastpath_analysis(
            baseline_systems=baseline_systems,
            fastpath_systems=fastpath_systems,
            repository_metadata=repository_metadata,
            target_improvement=args.target_improvement
        )
        logger.info("✅ Primary analysis completed")
        
        # Run cross-validation analysis
        logger.info("🔄 Running cross-validation stability analysis...")
        cv_results = run_cross_validation_analysis(
            framework, baseline_systems, fastpath_systems, args.cross_validation_folds
        )
        logger.info("✅ Cross-validation analysis completed")
        
        # Generate publication artifacts
        logger.info("📄 Generating publication-ready artifacts...")
        artifacts = framework.generate_publication_artifacts(
            analysis_results, args.output_dir / "artifacts"
        )
        logger.info(f"✅ Generated {len(artifacts)} publication artifacts")
        
        # Generate comprehensive report
        logger.info("📋 Generating comprehensive analysis report...")
        report_path = generate_comprehensive_report(
            analysis_results, cv_results, experimental_design, args.output_dir
        )
        logger.info("✅ Comprehensive report generated")
        
        # Save all results
        results_file = args.output_dir / "complete_statistical_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'experimental_design': experimental_design.__dict__,
                'analysis_results': analysis_results,
                'cross_validation_results': cv_results,
                'artifacts_generated': list(artifacts.keys())
            }, f, indent=2, default=str)
        
        # Print executive summary
        pub_assessment = analysis_results['publication_assessment']
        
        print("\n" + "="*80)
        print("EXECUTIVE SUMMARY - STATISTICAL VALIDATION RESULTS")
        print("="*80)
        
        print(f"📊 Overall Publication Readiness: {pub_assessment['overall_score']:.1f}/100")
        print(f"✅ Primary Hypothesis: {'SUPPORTED' if pub_assessment['primary_hypothesis_supported'] else 'NOT SUPPORTED'}")
        print(f"🔬 Statistical Rigor: {pub_assessment['statistical_rigor_score']:.1f}/100")
        print(f"⚡ Statistical Power: {'ADEQUATE' if pub_assessment['sample_size_adequate'] else 'INADEQUATE'}")
        print(f"📈 Effect Sizes: {'ADEQUATE' if pub_assessment['effect_size_adequate'] else 'INADEQUATE'}")
        
        # Show key metrics
        primary_comps = analysis_results['primary_comparisons']
        if primary_comps:
            print(f"\n📈 Key Performance Results:")
            for comp_name, comp_data in list(primary_comps.items())[:3]:  # Show first 3
                improvement = comp_data.get('improvement_percentage', 0)
                cohens_d = comp_data['effect_sizes']['cohens_d']['cohens_d']
                meets_target = comp_data.get('meets_target', False)
                target_icon = "✅" if meets_target else "❌"
                
                clean_name = comp_name.replace('_', ' ').title()
                print(f"  • {clean_name}: {improvement:.1f}% improvement, Cohen's d={cohens_d:.2f} {target_icon}")
        
        # Show cross-validation stability
        if cv_results and 'stability_metrics' in cv_results:
            consistency = cv_results['stability_metrics'].get('consistency_assessment', 'unknown')
            print(f"\n🔄 Cross-Validation Stability: {consistency.upper()}")
        
        # Show publication readiness
        venues = pub_assessment.get('suitable_venues', [])[:3]
        print(f"\n📄 Suitable Publication Venues: {', '.join(venues)}")
        print(f"🎯 Estimated Acceptance Probability: {pub_assessment.get('estimated_acceptance_probability', 0):.1%}")
        
        # Show file locations
        print(f"\n📁 Results Files:")
        print(f"  • Complete Results: {results_file}")
        print(f"  • Analysis Report: {report_path}")
        print(f"  • Publication Artifacts: {args.output_dir / 'artifacts'}")
        
        print("\n" + "="*80)
        print("STATISTICAL VALIDATION COMPLETED SUCCESSFULLY")
        print("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Statistical validation failed: {e}")
        logger.error(f"Error details: {type(e).__name__}")
        if args.verbose:
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
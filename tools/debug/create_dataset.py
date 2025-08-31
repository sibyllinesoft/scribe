#!/usr/bin/env python3
"""
PackRepo Dataset Builder CLI

Command-line interface for creating high-quality QA datasets that meet
all TODO.md requirements for rigorous evaluation of PackRepo's token
efficiency claims.

Usage Examples:
    # Create sample dataset for testing
    python create_dataset.py --sample --output ./datasets/sample_dataset

    # Create dataset from existing repositories
    python create_dataset.py --repos /path/to/repo1 /path/to/repo2 --output ./datasets/production

    # Validate existing dataset
    python create_dataset.py --validate ./datasets/existing_dataset.jsonl

    # Create TODO.md compliant dataset (automatic requirements)
    python create_dataset.py --todo-compliant --output ./datasets/compliant
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Add packrepo to path
sys.path.insert(0, str(Path(__file__).parent / "packrepo"))

from packrepo.evaluator.datasets.dataset_builder import (
    DatasetBuildOrchestrator, 
    DatasetBuildConfig,
    ComprehensiveDatasetBuilder
)
from packrepo.evaluator.datasets.schema import validate_dataset
from packrepo.evaluator.datasets.curator import CurationCriteria


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_sample_dataset(output_dir: Path, num_repos: int, questions_per_repo: int, verbose: bool):
    """Create sample dataset for testing."""
    print(f"🏗️  Creating sample dataset with {num_repos} repositories...")
    print(f"📍 Output directory: {output_dir}")
    
    result = DatasetBuildOrchestrator.build_from_sample_repos(
        output_dir=output_dir,
        num_repos=num_repos,
        questions_per_repo=questions_per_repo
    )
    
    return result


def create_dataset_from_repos(repo_paths: List[Path], output_dir: Path, config_file: Path, verbose: bool):
    """Create dataset from existing repositories."""
    print(f"🏗️  Creating dataset from {len(repo_paths)} repositories...")
    print(f"📍 Repositories: {[str(p) for p in repo_paths]}")
    print(f"📍 Output directory: {output_dir}")
    
    # Load config if provided
    config = None
    if config_file and config_file.exists():
        import json
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Create config from loaded data
            config = DatasetBuildConfig(
                output_dir=output_dir,
                min_questions=config_data.get('min_questions', 300),
                min_repositories=config_data.get('min_repositories', 5),
                questions_per_repo=config_data.get('questions_per_repo', 60),
                min_kappa_threshold=config_data.get('min_kappa_threshold', 0.6),
                annotation_sample_size=config_data.get('annotation_sample_size', 50)
            )
            print(f"📋 Using configuration from {config_file}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load config file {config_file}: {e}")
            print("Using default configuration")
    
    result = DatasetBuildOrchestrator.build_from_existing_repos(
        repository_paths=repo_paths,
        output_dir=output_dir,
        config=config
    )
    
    return result


def validate_existing_dataset(dataset_path: Path, verbose: bool):
    """Validate existing dataset."""
    print(f"🔍 Validating dataset: {dataset_path}")
    
    validation_result = DatasetBuildOrchestrator.validate_existing_dataset(dataset_path)
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    if validation_result.get('overall_valid', False):
        print("✅ Dataset is VALID")
    else:
        print("❌ Dataset has ISSUES")
    
    # Schema validation
    schema_valid = validation_result.get('schema_valid', False)
    print(f"Schema Valid: {'✅' if schema_valid else '❌'}")
    
    schema_errors = validation_result.get('schema_errors', [])
    if schema_errors:
        print("Schema Errors:")
        for error in schema_errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(schema_errors) > 5:
            print(f"  ... and {len(schema_errors) - 5} more errors")
    
    # Distribution analysis
    distribution = validation_result.get('distribution', {})
    if distribution:
        print(f"\nDistribution Analysis:")
        print(f"  Total Questions: {distribution.get('total_questions', 0)}")
        print(f"  Total Repositories: {distribution.get('total_repos', 0)}")
        
        requirements = distribution.get('meets_requirements', {})
        print(f"  Meets Requirements:")
        print(f"    ≥300 Questions: {'✅' if requirements.get('min_questions', False) else '❌'}")
        print(f"    ≥5 Repositories: {'✅' if requirements.get('min_repos', False) else '❌'}")
        print(f"    Difficulty Distribution: {'✅' if requirements.get('difficulty_distribution', False) else '❌'}")
        
        # Show difficulty percentages
        difficulty_pct = distribution.get('difficulty_percentages', {})
        if difficulty_pct:
            print(f"  Difficulty Distribution:")
            for difficulty, pct in difficulty_pct.items():
                target_ranges = {
                    'easy': '15-25%',
                    'medium': '40-60%', 
                    'hard': '25-35%'
                }
                target = target_ranges.get(difficulty, 'N/A')
                print(f"    {difficulty.title()}: {pct:.1%} (target: {target})")
    
    return validation_result


def create_todo_compliant_dataset(output_dir: Path, use_samples: bool, verbose: bool):
    """Create dataset that meets all TODO.md requirements."""
    print("🎯 Creating TODO.md compliant dataset...")
    print("📋 Requirements:")
    print("   - ≥ 300 questions across ≥ 5 repositories")
    print("   - Inter-annotator agreement κ≥0.6 on 50-question audit")
    print("   - Proper ground truth with verifiable references")
    print("   - Diverse programming languages and domains")
    print(f"📍 Output directory: {output_dir}")
    
    result = DatasetBuildOrchestrator.create_todo_compliant_dataset(
        output_dir=output_dir,
        use_samples=use_samples
    )
    
    return result


def print_build_results(result):
    """Print comprehensive build results."""
    print("\n" + "="*60)
    print("DATASET BUILD RESULTS")
    print("="*60)
    
    # Overall status
    if result.success:
        print("🎉 BUILD SUCCESSFUL")
    else:
        print("❌ BUILD FAILED")
    
    print(f"📊 Dataset: {result.dataset_path}")
    print(f"📊 Total Questions: {result.total_questions}")
    print(f"📊 Repository Count: {result.repository_count}")
    print(f"⏱️  Build Time: {result.build_time_seconds:.1f}s")
    
    # TODO.md compliance
    if result.meets_todo_requirements:
        print("✅ MEETS ALL TODO.md REQUIREMENTS")
    else:
        print("❌ Does not meet TODO.md requirements")
    
    # Quality metrics
    if result.quality_metrics:
        qm = result.quality_metrics
        print(f"\n📊 Quality Metrics:")
        print(f"   Validation Pass Rate: {qm.get('validation_pass_rate', 0):.1%}")
        print(f"   Mean Confidence Score: {qm.get('mean_confidence_score', 0):.3f}")
        print(f"   Consistency Score: {qm.get('consistency_score', 0):.3f}")
        print(f"   Completeness Score: {qm.get('completeness_score', 0):.3f}")
        
        # Distribution summary
        difficulty_dist = qm.get('difficulty_distribution', {})
        if difficulty_dist:
            print(f"   Difficulty Distribution:")
            for diff, pct in difficulty_dist.items():
                print(f"     {diff.title()}: {pct:.1%}")
    
    # Inter-annotator agreement
    if result.inter_annotator_agreement:
        iaa = result.inter_annotator_agreement
        print(f"\n📊 Inter-Annotator Agreement:")
        print(f"   Cohen's κ: {iaa.get('cohens_kappa', 0):.3f}")
        threshold_met = iaa.get('meets_threshold', False)
        print(f"   Meets κ≥0.6 Threshold: {'✅' if threshold_met else '❌'}")
        print(f"   Sample Size: {iaa.get('sample_size', 0)}")
        print(f"   Agreement Percentage: {iaa.get('agreement_percentage', 0):.1%}")
    
    # Issues and recommendations
    if result.validation_errors:
        print(f"\n⚠️  Validation Issues ({len(result.validation_errors)}):")
        for error in result.validation_errors[:3]:  # Show first 3
            print(f"   - {error}")
        if len(result.validation_errors) > 3:
            print(f"   ... and {len(result.validation_errors) - 3} more issues")
    
    # Reports location
    if result.dataset_path:
        reports_dir = result.dataset_path.parent / "reports"
        print(f"\n📋 Detailed reports available in: {reports_dir}")
        print(f"📋 Repository configs in: {result.dataset_path.parent / 'repo_configs'}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PackRepo Dataset Builder - Create high-quality QA datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Main command group
    command_group = parser.add_mutually_exclusive_group(required=True)
    
    command_group.add_argument(
        '--sample',
        action='store_true',
        help='Create dataset from sample repositories (for testing)'
    )
    
    command_group.add_argument(
        '--repos',
        nargs='+',
        type=Path,
        help='Create dataset from existing repository paths'
    )
    
    command_group.add_argument(
        '--validate',
        type=Path,
        help='Validate existing dataset file'
    )
    
    command_group.add_argument(
        '--todo-compliant',
        action='store_true',
        help='Create dataset that meets all TODO.md requirements'
    )
    
    # Configuration options
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('./datasets'),
        help='Output directory for dataset and reports (default: ./datasets)'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='JSON configuration file for dataset building'
    )
    
    # Sample dataset options
    parser.add_argument(
        '--num-repos',
        type=int,
        default=6,
        help='Number of sample repositories to create (default: 6)'
    )
    
    parser.add_argument(
        '--questions-per-repo',
        type=int,
        default=55,
        help='Questions per repository (default: 55)'
    )
    
    # General options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--use-samples',
        action='store_true',
        help='Use sample repositories for TODO compliant dataset (for testing)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        if args.sample:
            # Create sample dataset
            result = create_sample_dataset(
                output_dir=args.output,
                num_repos=args.num_repos,
                questions_per_repo=args.questions_per_repo,
                verbose=args.verbose
            )
            print_build_results(result)
            return 0 if result.success else 1
            
        elif args.repos:
            # Create dataset from existing repos
            # Validate repository paths
            invalid_paths = [p for p in args.repos if not p.exists()]
            if invalid_paths:
                print(f"❌ Invalid repository paths: {invalid_paths}")
                return 1
            
            result = create_dataset_from_repos(
                repo_paths=args.repos,
                output_dir=args.output,
                config_file=args.config,
                verbose=args.verbose
            )
            print_build_results(result)
            return 0 if result.success else 1
            
        elif args.validate:
            # Validate existing dataset
            if not args.validate.exists():
                print(f"❌ Dataset file not found: {args.validate}")
                return 1
            
            validation_result = validate_existing_dataset(
                dataset_path=args.validate,
                verbose=args.verbose
            )
            return 0 if validation_result.get('overall_valid', False) else 1
            
        elif args.todo_compliant:
            # Create TODO.md compliant dataset
            result = create_todo_compliant_dataset(
                output_dir=args.output,
                use_samples=args.use_samples,
                verbose=args.verbose
            )
            print_build_results(result)
            return 0 if result.success and result.meets_todo_requirements else 1
    
    except KeyboardInterrupt:
        print("\n🛑 Build interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
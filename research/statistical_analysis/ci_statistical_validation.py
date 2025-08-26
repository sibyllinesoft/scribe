#!/usr/bin/env python3
"""
CI/CD Statistical Validation Script for PackRepo FastPath V2

This script provides CI/CD pipeline integration for automated statistical validation
and promotion decisions based on comprehensive quality analysis.

Usage in CI/CD:
    # GitHub Actions example
    - name: Statistical Validation
      run: python ci_statistical_validation.py --baseline-file baseline.json --test-results test_results.json
      env:
        FASTPATH_VALIDATION_THRESHOLD: 13.0
        PERFORMANCE_REGRESSION_LIMIT: 10.0

Exit Codes:
    0: All validation criteria met, approve for promotion
    1: Validation criteria not met, block promotion  
    2: Invalid configuration or missing data
    3: Infrastructure/dependency issues
    4: Unexpected error or interruption
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Set up logging for CI/CD visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import our analytics runner
sys.path.insert(0, str(Path(__file__).parent))
from run_comprehensive_analytics import ComprehensiveAnalyticsRunner


class CIValidationOrchestrator:
    """Orchestrates statistical validation for CI/CD pipelines."""
    
    def __init__(self):
        self.improvement_threshold = float(os.getenv('FASTPATH_VALIDATION_THRESHOLD', '13.0'))
        self.regression_limit = float(os.getenv('PERFORMANCE_REGRESSION_LIMIT', '10.0'))
        self.output_dir = Path(os.getenv('CI_ANALYTICS_OUTPUT_DIR', './ci_analytics_results'))
        
        logger.info("🏭 CI Statistical Validation Orchestrator initialized")
        logger.info(f"📊 Improvement threshold: ≥{self.improvement_threshold}%")
        logger.info(f"⚡ Regression limit: ≤{self.regression_limit}%")
        
    async def validate_fastpath_promotion(self, baseline_file: Path, 
                                        test_results_file: Path) -> int:
        """
        Execute comprehensive statistical validation for FastPath promotion.
        
        Returns:
            0: Approve promotion
            1: Block promotion
            2: Invalid configuration
            3: Infrastructure issues
            4: Unexpected error
        """
        
        try:
            # Validate input files
            if not baseline_file.exists():
                logger.error(f"❌ Baseline file not found: {baseline_file}")
                return 2
                
            if not test_results_file.exists():
                logger.error(f"❌ Test results file not found: {test_results_file}")
                return 2
            
            # Load data files
            logger.info(f"📥 Loading baseline data from {baseline_file}")
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
            
            logger.info(f"📥 Loading test results from {test_results_file}")
            with open(test_results_file, 'r') as f:
                test_data = json.load(f)
            
            # Initialize analytics runner with CI-specific configuration
            runner = ComprehensiveAnalyticsRunner(output_dir=self.output_dir)
            
            # Override thresholds with CI configuration
            runner.min_improvement_threshold = self.improvement_threshold
            runner.max_regression_threshold = self.regression_limit
            
            logger.info("🔬 Executing comprehensive statistical analysis...")
            
            # Run full analysis
            results = await runner.run_full_analysis(baseline_data, test_data)
            
            # Extract promotion decision
            promotion_decision = results['promotion_decision']
            
            # Log detailed results for CI visibility
            self._log_validation_results(results)
            
            # Generate CI artifacts
            await self._generate_ci_artifacts(results)
            
            # Return appropriate exit code
            if promotion_decision['promote']:
                logger.info("✅ VALIDATION PASSED - FastPath V2 approved for promotion")
                return 0
            else:
                logger.warning("❌ VALIDATION FAILED - FastPath V2 promotion blocked")
                self._log_failure_reasons(promotion_decision)
                return 1
                
        except FileNotFoundError as e:
            logger.error(f"❌ Required file not found: {e}")
            return 2
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in data files: {e}")
            return 2
        except Exception as e:
            logger.error(f"❌ Unexpected validation error: {e}")
            logger.exception("Full traceback:")
            return 4
    
    def _log_validation_results(self, results: Dict[str, Any]) -> None:
        """Log detailed validation results for CI visibility."""
        
        report = results['report']
        
        logger.info("📊 STATISTICAL VALIDATION RESULTS")
        logger.info("=" * 50)
        
        # Executive summary
        summary = report['executive_summary']
        logger.info(f"QA Improvement: {summary['qa_improvement_percentage']:.2f}%")
        logger.info(f"Confidence Interval: [{summary['confidence_interval'][0]:.2f}%, {summary['confidence_interval'][1]:.2f}%]")
        logger.info(f"Evidence Strength: {summary['evidence_strength']}")
        
        # Analysis results
        analysis = report['statistical_analysis']
        
        # Bootstrap analysis
        bootstrap = analysis['bca_bootstrap']
        logger.info(f"📈 Bootstrap Analysis: {'✅ PASS' if bootstrap['success_criterion_met'] else '❌ FAIL'}")
        
        # Performance regression
        regression = analysis['performance_regression']
        logger.info(f"⚡ Performance Regression: {'✅ PASS' if regression['overall_regression_status']['all_thresholds_met'] else '❌ FAIL'}")
        
        # Quality gates
        quality = analysis['quality_gates']
        logger.info(f"🛡️ Quality Gates: {'✅ PASS' if quality['overall_quality_gates']['all_passed'] else '❌ FAIL'}")
        
        # Category analysis
        category = analysis['category_analysis']
        logger.info(f"📊 Category Analysis: {'✅ PASS' if category['overall_category_status']['no_degradation'] else '❌ FAIL'}")
    
    def _log_failure_reasons(self, promotion_decision: Dict[str, Any]) -> None:
        """Log specific reasons for validation failure."""
        
        logger.warning("🔍 FAILURE ANALYSIS")
        logger.warning("-" * 30)
        
        criteria = promotion_decision['criteria_results']
        
        for criterion, passed in criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.warning(f"{criterion.replace('_', ' ').title()}: {status}")
    
    async def _generate_ci_artifacts(self, results: Dict[str, Any]) -> None:
        """Generate CI artifacts for reporting and debugging."""
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Custom JSON encoder to handle numpy types
        def json_serializer(obj):
            """Custom JSON serializer for numpy types."""
            if hasattr(obj, 'item'):
                return obj.item()  # Convert numpy scalar to Python scalar
            return str(obj)
        
        # Save full results
        full_results_file = self.output_dir / f"ci_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(full_results_file, 'w') as f:
            json.dump(results, f, indent=2, default=json_serializer)
        
        # Create promotion status file for downstream CI steps
        promotion_status = {
            'timestamp': datetime.now().isoformat(),
            'promote': bool(results['promotion_decision']['promote']),  # Ensure Python bool
            'decision': str(results['promotion_decision']['decision']),
            'recommendation': str(results['promotion_decision']['recommendation'])
        }
        
        status_file = self.output_dir / 'promotion_status.json'
        with open(status_file, 'w') as f:
            json.dump(promotion_status, f, indent=2)
        
        # Create summary for GitHub Actions or other CI systems
        summary_file = self.output_dir / 'validation_summary.md'
        await self._create_markdown_summary(results, summary_file)
        
        logger.info(f"📁 CI artifacts generated in {self.output_dir}")
    
    async def _create_markdown_summary(self, results: Dict[str, Any], output_file: Path) -> None:
        """Create a markdown summary for GitHub Actions or other CI reporting."""
        
        report = results['report']
        decision = results['promotion_decision']
        
        summary = f"""# FastPath V2 Statistical Validation Results

## Promotion Decision: {decision['decision']}

{decision['recommendation']}

## Executive Summary

- **QA Improvement**: {report['executive_summary']['qa_improvement_percentage']:.2f}%
- **Confidence Interval**: [{report['executive_summary']['confidence_interval'][0]:.2f}%, {report['executive_summary']['confidence_interval'][1]:.2f}%]
- **Evidence Strength**: {report['executive_summary']['evidence_strength']}

## Validation Results

| Criterion | Status | Details |
|-----------|--------|---------|
| Statistical Improvement | {'✅ PASS' if decision['criteria_results']['statistical_improvement'] else '❌ FAIL'} | ≥13% improvement with 95% confidence |
| Performance Regression | {'✅ PASS' if decision['criteria_results']['performance_regression'] else '❌ FAIL'} | ≤10% latency/memory regression |
| Quality Gates | {'✅ PASS' if decision['criteria_results']['quality_gates'] else '❌ FAIL'} | Mutation score, coverage, security |
| Category Analysis | {'✅ PASS' if decision['criteria_results']['category_analysis'] else '❌ FAIL'} | No >5 point category degradation |

## Analysis Details

### Bootstrap Analysis
- Bootstrap Samples: {report['statistical_analysis']['bca_bootstrap']['bootstrap_samples']:,}
- Confidence Level: {report['statistical_analysis']['bca_bootstrap']['confidence_level']}
- Success Criterion: {'Met' if report['statistical_analysis']['bca_bootstrap']['success_criterion_met'] else 'Not Met'}

Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(output_file, 'w') as f:
            f.write(summary)


async def main():
    """Main entry point for CI statistical validation."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CI/CD Statistical Validation for PackRepo FastPath V2",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--baseline-file',
        type=Path,
        required=True,
        help='Path to baseline performance metrics JSON file'
    )
    
    parser.add_argument(
        '--test-results',
        type=Path,
        required=True,
        help='Path to FastPath test results JSON file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Override output directory for CI artifacts'
    )
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = CIValidationOrchestrator()
    
    if args.output_dir:
        orchestrator.output_dir = args.output_dir
    
    # Run validation
    exit_code = await orchestrator.validate_fastpath_promotion(
        args.baseline_file, 
        args.test_results
    )
    
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
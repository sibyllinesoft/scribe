#!/usr/bin/env python3
"""
PackRepo Analytics Dashboard

Comprehensive analytics dashboard for monitoring and optimizing QA dataset
quality, generation performance, and evaluation effectiveness. Provides
real-time insights and actionable recommendations for continuous improvement.

Features:
- Real-time quality monitoring
- Performance optimization insights
- Statistical validation tracking
- Predictive analytics for dataset effectiveness
- Automated reporting and alerting
- Comparative benchmarking
- ROI analysis for dataset investment
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import time

# Add packrepo to path
sys.path.insert(0, str(Path(__file__).parent / "packrepo"))

from packrepo.evaluator.datasets.analytics import DatasetAnalytics, MetricType
from packrepo.evaluator.datasets.dataset_builder import DatasetBuildOrchestrator
from packrepo.evaluator.datasets.schema import QuestionItem, RepositoryMetadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class AnalyticsDashboard:
    """
    Comprehensive analytics dashboard for PackRepo datasets.
    
    Provides strategic analytics capabilities:
    - Real-time dataset quality monitoring
    - Performance optimization recommendations  
    - ROI analysis for dataset creation investment
    - Predictive analytics for evaluation success
    - Automated alerting for quality degradation
    - Competitive benchmarking
    """
    
    def __init__(self, analytics_db_path: Optional[Path] = None):
        """Initialize analytics dashboard."""
        self.analytics = DatasetAnalytics(analytics_db_path)
        self.dashboard_start_time = datetime.now()
        
        logger.info("🚀 PackRepo Analytics Dashboard initialized")
        logger.info(f"📊 Analytics database: {self.analytics.analytics_db_path}")
        
    def run_comprehensive_analysis(self, dataset_path: Path) -> Dict[str, any]:
        """
        Run comprehensive analysis on a dataset.
        
        Returns:
            Dictionary containing all analysis results and insights
        """
        logger.info("🔍 Starting comprehensive dataset analysis...")
        analysis_start = time.time()
        
        # Load dataset
        questions, repo_metadata = self._load_dataset(dataset_path)
        if not questions:
            logger.error(f"❌ No questions found in dataset: {dataset_path}")
            return {"success": False, "error": "Empty dataset"}
        
        logger.info(f"📊 Loaded {len(questions)} questions from {len(set(q.repo_id for q in questions))} repositories")
        
        # Generate comprehensive analytics report
        generation_time = self._estimate_generation_time(len(questions))
        report = self.analytics.generate_comprehensive_report(
            dataset_path=dataset_path,
            questions=questions,
            repository_metadata=repo_metadata,
            generation_time=generation_time
        )
        
        analysis_duration = time.time() - analysis_start
        
        # Create strategic insights
        strategic_insights = self._generate_strategic_insights(report, questions, repo_metadata)
        
        # Calculate ROI metrics
        roi_analysis = self._calculate_roi_analysis(report, len(questions), generation_time)
        
        # Generate recommendations
        strategic_recommendations = self._generate_strategic_recommendations(report, strategic_insights)
        
        # Compile comprehensive results
        results = {
            "success": True,
            "analysis_duration_seconds": analysis_duration,
            "dataset_summary": {
                "total_questions": len(questions),
                "total_repositories": len(set(q.repo_id for q in questions)),
                "quality_score": report.overall_quality_score,
                "meets_todo_requirements": self._check_todo_compliance(report, questions)
            },
            "quality_metrics": {
                "overall_quality": report.overall_quality_score,
                "completeness": report.completeness_score,
                "consistency": report.consistency_score,
                "validation_pass_rate": report.validation_pass_rate
            },
            "distribution_analysis": {
                "difficulty_distribution": report.difficulty_distribution,
                "category_distribution": report.category_distribution,
                "language_coverage": report.language_coverage,
                "domain_coverage": report.domain_coverage
            },
            "performance_metrics": report.generation_efficiency,
            "strategic_insights": strategic_insights,
            "roi_analysis": roi_analysis,
            "recommendations": strategic_recommendations,
            "trend_analysis": report.trend_analysis,
            "benchmark_comparison": report.benchmark_comparison,
            "improvement_opportunities": report.improvement_opportunities
        }
        
        logger.info(f"✅ Comprehensive analysis completed in {analysis_duration:.1f}s")
        logger.info(f"🎯 Overall quality score: {report.overall_quality_score:.1%}")
        logger.info(f"📈 Generated {len(strategic_insights)} strategic insights")
        
        return results
    
    def _load_dataset(self, dataset_path: Path) -> tuple[List[QuestionItem], List[RepositoryMetadata]]:
        """Load dataset from JSON Lines file."""
        questions = []
        repo_metadata = []
        
        try:
            if not dataset_path.exists():
                logger.warning(f"Dataset file not found: {dataset_path}")
                return questions, repo_metadata
            
            with open(dataset_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        
                        # Create QuestionItem (simplified conversion)
                        question = QuestionItem(
                            repo_id=data.get('repo_id', 'unknown'),
                            qid=data.get('qid', f'q_{line_num}'),
                            question=data.get('question', ''),
                            gold=data.get('gold', {}),
                            category=data.get('category', 'function_behavior'),
                            difficulty=data.get('difficulty', 'medium'),
                            evaluation_type=data.get('evaluation_type', 'semantic'),
                            pack_budget=data.get('pack_budget', 50000),
                            rubric=data.get('rubric')
                        )
                        questions.append(question)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
        
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
        
        return questions, repo_metadata
    
    def _estimate_generation_time(self, num_questions: int) -> float:
        """Estimate generation time based on question count."""
        # Rough estimate: 0.5 seconds per question for analysis and generation
        return num_questions * 0.5
    
    def _generate_strategic_insights(self, report, questions: List[QuestionItem], 
                                   repo_metadata: List[RepositoryMetadata]) -> List[Dict[str, str]]:
        """Generate strategic insights for decision making."""
        insights = []
        
        # Quality insights
        quality_score = report.overall_quality_score
        if quality_score >= 0.85:
            insights.append({
                "category": "Quality Excellence",
                "insight": f"Dataset achieves research-grade quality ({quality_score:.1%})",
                "impact": "High confidence in evaluation results",
                "action": "Ready for production evaluation and benchmarking"
            })
        elif quality_score >= 0.70:
            insights.append({
                "category": "Quality Optimization",
                "insight": f"Dataset quality is good ({quality_score:.1%}) with improvement potential",
                "impact": "Reliable evaluation with room for enhancement",
                "action": "Focus on top improvement opportunities for maximum ROI"
            })
        else:
            insights.append({
                "category": "Quality Risk",
                "insight": f"Dataset quality below industry standards ({quality_score:.1%})",
                "impact": "Evaluation results may not be reliable",
                "action": "Immediate quality improvement required before production use"
            })
        
        # Scale insights
        total_questions = len(questions)
        unique_repos = len(set(q.repo_id for q in questions))
        
        if total_questions >= 500 and unique_repos >= 10:
            insights.append({
                "category": "Scale Achievement",
                "insight": f"Large-scale dataset ({total_questions} questions, {unique_repos} repos)",
                "impact": "Enables robust statistical analysis and generalizability",
                "action": "Leverage scale for comprehensive evaluation studies"
            })
        elif total_questions >= 300 and unique_repos >= 5:
            insights.append({
                "category": "Scale Adequacy",
                "insight": f"Dataset meets minimum scale requirements ({total_questions} questions, {unique_repos} repos)",
                "impact": "Sufficient for basic evaluation but limited statistical power",
                "action": "Consider expansion for more robust statistical confidence"
            })
        else:
            insights.append({
                "category": "Scale Limitation",
                "insight": f"Dataset below minimum scale ({total_questions} questions, {unique_repos} repos)",
                "impact": "Limited statistical power and generalizability",
                "action": "Expand dataset size before conducting evaluations"
            })
        
        # Diversity insights
        language_count = len(report.language_coverage) if report.language_coverage else 0
        domain_count = len(report.domain_coverage) if report.domain_coverage else 0
        
        if language_count >= 5 and domain_count >= 4:
            insights.append({
                "category": "Diversity Strength",
                "insight": f"Excellent diversity ({language_count} languages, {domain_count} domains)",
                "impact": "Results generalizable across programming ecosystems",
                "action": "Maintain diversity in future dataset expansions"
            })
        elif language_count >= 3 and domain_count >= 3:
            insights.append({
                "category": "Diversity Moderate",
                "insight": f"Good diversity ({language_count} languages, {domain_count} domains)",
                "impact": "Results applicable to common programming scenarios",
                "action": "Consider expanding to niche languages/domains for broader coverage"
            })
        
        # Performance insights
        if report.generation_efficiency:
            efficiency = report.generation_efficiency.get('efficiency_score', 0.0)
            if efficiency >= 0.8:
                insights.append({
                    "category": "Generation Efficiency",
                    "insight": f"Highly efficient generation pipeline ({efficiency:.1%})",
                    "impact": "Cost-effective scaling and rapid iteration capability",
                    "action": "Document best practices for team knowledge sharing"
                })
            elif efficiency < 0.5:
                insights.append({
                    "category": "Performance Bottleneck",
                    "insight": f"Generation pipeline needs optimization ({efficiency:.1%})",
                    "impact": "Higher costs and slower iteration cycles",
                    "action": "Implement performance optimization recommendations"
                })
        
        # Trend insights (if historical data available)
        improving_trends = [name for name, direction in report.trend_analysis.items() 
                          if direction.value == "improving"]
        if improving_trends:
            insights.append({
                "category": "Positive Momentum",
                "insight": f"Quality trends improving ({len(improving_trends)} metrics)",
                "impact": "Continuous improvement process is working",
                "action": "Maintain current optimization efforts and expand successful practices"
            })
        
        return insights
    
    def _calculate_roi_analysis(self, report, num_questions: int, generation_time: float) -> Dict[str, float]:
        """Calculate return on investment analysis."""
        # Cost estimation (simplified)
        cost_per_question = 2.0  # Estimated $2 per question (time + compute)
        total_cost = num_questions * cost_per_question
        
        # Quality multiplier
        quality_multiplier = report.overall_quality_score * 2  # Quality boosts value
        
        # Value estimation
        base_value_per_question = 5.0  # Base value $5 per question
        total_value = num_questions * base_value_per_question * quality_multiplier
        
        # Efficiency factor
        if report.generation_efficiency:
            efficiency_factor = report.generation_efficiency.get('efficiency_score', 0.5)
        else:
            efficiency_factor = 0.5
        
        roi = ((total_value - total_cost) / total_cost) * 100 if total_cost > 0 else 0
        
        return {
            "total_investment_usd": total_cost,
            "estimated_value_usd": total_value,
            "roi_percentage": roi,
            "quality_multiplier": quality_multiplier,
            "efficiency_factor": efficiency_factor,
            "cost_per_question": cost_per_question,
            "value_per_question": base_value_per_question * quality_multiplier,
            "payback_period_days": 30 if roi > 100 else 90  # Simplified estimate
        }
    
    def _generate_strategic_recommendations(self, report, strategic_insights: List[Dict]) -> List[Dict[str, str]]:
        """Generate strategic recommendations based on analysis."""
        recommendations = []
        
        # Quality-based strategic recommendations
        quality_score = report.overall_quality_score
        if quality_score < 0.8:
            recommendations.append({
                "priority": "High",
                "category": "Quality Investment",
                "recommendation": "Invest in quality improvement pipeline before scaling",
                "rationale": f"Current quality ({quality_score:.1%}) limits evaluation reliability",
                "estimated_effort": "2-3 weeks",
                "expected_impact": "+20-30% quality score improvement"
            })
        
        # Scale recommendations
        quality_risks = [i for i in strategic_insights if i["category"] == "Quality Risk"]
        if quality_risks:
            recommendations.append({
                "priority": "Critical",
                "category": "Risk Mitigation",
                "recommendation": "Halt production usage until quality issues are resolved",
                "rationale": "Low quality datasets can produce misleading evaluation results",
                "estimated_effort": "1-2 weeks",
                "expected_impact": "Prevent evaluation reliability issues"
            })
        
        # Efficiency recommendations
        if report.generation_efficiency:
            efficiency = report.generation_efficiency.get('efficiency_score', 0.0)
            if efficiency < 0.6:
                recommendations.append({
                    "priority": "Medium",
                    "category": "Operational Efficiency",
                    "recommendation": "Implement parallel processing and caching optimizations",
                    "rationale": f"Current efficiency ({efficiency:.1%}) increases costs and delays",
                    "estimated_effort": "1 week",
                    "expected_impact": "+40-60% generation speed improvement"
                })
        
        # Diversity recommendations
        if report.language_coverage and len(report.language_coverage) < 4:
            recommendations.append({
                "priority": "Medium",
                "category": "Coverage Enhancement",
                "recommendation": "Expand programming language coverage to include Rust, Go, and Kotlin",
                "rationale": "Limited language diversity reduces evaluation generalizability",
                "estimated_effort": "2 weeks",
                "expected_impact": "+25% broader applicability"
            })
        
        # Innovation recommendations
        recommendations.append({
            "priority": "Low",
            "category": "Innovation Opportunity",
            "recommendation": "Explore AI-assisted question generation for novel question types",
            "rationale": "Current question templates may miss emerging code patterns",
            "estimated_effort": "3-4 weeks",
            "expected_impact": "Unlock new evaluation dimensions"
        })
        
        return recommendations
    
    def _check_todo_compliance(self, report, questions: List[QuestionItem]) -> Dict[str, bool]:
        """Check compliance with TODO.md requirements."""
        compliance = {}
        
        # ≥ 300 questions
        compliance["min_questions"] = len(questions) >= 300
        
        # ≥ 5 repositories  
        unique_repos = len(set(q.repo_id for q in questions))
        compliance["min_repositories"] = unique_repos >= 5
        
        # Inter-annotator agreement κ≥0.6
        if report.agreement_metrics and 'simulated_kappa' in report.agreement_metrics:
            compliance["min_kappa"] = report.agreement_metrics['simulated_kappa'] >= 0.6
        else:
            compliance["min_kappa"] = False
        
        # Quality standards
        compliance["quality_threshold"] = report.overall_quality_score >= 0.7
        
        # Difficulty distribution (approximately 20%, 50%, 30%)
        if report.difficulty_distribution:
            easy_pct = report.difficulty_distribution.get('easy', 0.0)
            medium_pct = report.difficulty_distribution.get('medium', 0.0)
            hard_pct = report.difficulty_distribution.get('hard', 0.0)
            
            # Allow ±10% tolerance
            compliance["difficulty_distribution"] = (
                0.1 <= easy_pct <= 0.3 and
                0.4 <= medium_pct <= 0.6 and
                0.2 <= hard_pct <= 0.4
            )
        else:
            compliance["difficulty_distribution"] = False
        
        return compliance
    
    def generate_executive_summary(self, analysis_results: Dict[str, any]) -> str:
        """Generate executive summary for stakeholders."""
        if not analysis_results.get("success"):
            return "❌ **Analysis Failed**: Unable to process dataset."
        
        summary = []
        
        # Header
        summary.append("# 📊 PackRepo QA Dataset Analytics Summary")
        summary.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        # Key metrics
        dataset_info = analysis_results["dataset_summary"]
        summary.append("## 🎯 Key Performance Indicators")
        summary.append(f"- **Overall Quality Score**: {dataset_info['quality_score']:.1%}")
        summary.append(f"- **Dataset Size**: {dataset_info['total_questions']} questions across {dataset_info['total_repositories']} repositories")
        summary.append(f"- **TODO.md Compliance**: {'✅ COMPLIANT' if dataset_info['meets_todo_requirements'] else '❌ NON-COMPLIANT'}")
        
        # ROI analysis
        roi_data = analysis_results["roi_analysis"]
        summary.append(f"- **ROI**: {roi_data['roi_percentage']:.1f}% return on investment")
        summary.append(f"- **Total Investment**: ${roi_data['total_investment_usd']:,.0f}")
        summary.append("")
        
        # Strategic insights
        insights = analysis_results["strategic_insights"]
        if insights:
            summary.append("## 💡 Strategic Insights")
            for insight in insights[:3]:  # Top 3 insights
                summary.append(f"- **{insight['category']}**: {insight['insight']}")
                summary.append(f"  - *Impact*: {insight['impact']}")
            summary.append("")
        
        # Top recommendations
        recommendations = analysis_results["recommendations"]
        if recommendations:
            summary.append("## 🚀 Top Recommendations")
            high_priority = [r for r in recommendations if r["priority"] == "High" or r["priority"] == "Critical"]
            for rec in high_priority[:2]:  # Top 2 high priority
                summary.append(f"- **{rec['category']}** ({rec['priority']} Priority)")
                summary.append(f"  - {rec['recommendation']}")
                summary.append(f"  - *Expected Impact*: {rec['expected_impact']}")
            summary.append("")
        
        # Quality breakdown
        quality_metrics = analysis_results["quality_metrics"]
        summary.append("## 📈 Quality Breakdown")
        summary.append(f"- **Completeness**: {quality_metrics.get('completeness', 0):.1%}")
        summary.append(f"- **Consistency**: {quality_metrics.get('consistency', 0):.1%}")
        summary.append(f"- **Validation Pass Rate**: {quality_metrics.get('validation_pass_rate', 0):.1%}")
        summary.append("")
        
        # Next steps
        summary.append("## 🎯 Immediate Next Steps")
        critical_recs = [r for r in recommendations if r["priority"] == "Critical"]
        if critical_recs:
            summary.append("**Critical Actions Required:**")
            for rec in critical_recs:
                summary.append(f"1. {rec['recommendation']} (Est. {rec['estimated_effort']})")
        else:
            summary.append("**Continue Current Optimization:**")
            summary.append("1. Monitor quality metrics for continued improvement")
            summary.append("2. Expand dataset scale when quality targets are met")
            summary.append("3. Implement efficiency optimizations for cost reduction")
        
        return "\n".join(summary)
    
    def export_results(self, analysis_results: Dict[str, any], output_dir: Path):
        """Export comprehensive analysis results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export detailed JSON report
        with open(output_dir / "comprehensive_analysis.json", 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Export executive summary
        executive_summary = self.generate_executive_summary(analysis_results)
        with open(output_dir / "executive_summary.md", 'w') as f:
            f.write(executive_summary)
        
        # Export recommendations CSV
        recommendations = analysis_results.get("recommendations", [])
        if recommendations:
            import csv
            with open(output_dir / "strategic_recommendations.csv", 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=recommendations[0].keys())
                writer.writeheader()
                writer.writerows(recommendations)
        
        logger.info(f"📁 Analysis results exported to {output_dir}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("PACKREPO ANALYTICS DASHBOARD SUMMARY")
        print("="*80)
        print(executive_summary)
        print("="*80)


def main():
    """Main dashboard execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PackRepo Analytics Dashboard - Strategic Dataset Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'dataset_path',
        type=Path,
        help='Path to QA dataset JSON Lines file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./analytics_reports'),
        help='Output directory for analysis reports (default: ./analytics_reports)'
    )
    
    parser.add_argument(
        '--analytics-db',
        type=Path,
        help='Path to analytics database (default: auto-generated)'
    )
    
    parser.add_argument(
        '--generate-visuals',
        action='store_true',
        help='Generate visualization charts (requires matplotlib)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize dashboard
        dashboard = AnalyticsDashboard(args.analytics_db)
        
        # Run comprehensive analysis
        print("🚀 Starting PackRepo Analytics Dashboard...")
        print(f"📊 Analyzing dataset: {args.dataset_path}")
        
        results = dashboard.run_comprehensive_analysis(args.dataset_path)
        
        if not results["success"]:
            print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
            return 1
        
        # Export results
        dashboard.export_results(results, args.output_dir)
        
        # Generate visualizations if requested
        if args.generate_visuals:
            try:
                # Create analytics report for visualization
                from packrepo.evaluator.datasets.analytics import AnalyticsReport
                # This would require converting results back to AnalyticsReport format
                print("📈 Visualization generation would be implemented here")
            except ImportError:
                print("⚠️  Visualization libraries not available. Install matplotlib and seaborn.")
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Reports saved to: {args.output_dir}")
        print(f"📊 Overall Quality Score: {results['dataset_summary']['quality_score']:.1%}")
        print(f"💰 ROI: {results['roi_analysis']['roi_percentage']:.1f}%")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Dashboard error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
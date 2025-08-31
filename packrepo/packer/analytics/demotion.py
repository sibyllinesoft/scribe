"""
V3 Demotion Analytics System

Provides comprehensive analysis and reporting for demotion patterns,
oscillation trends, and system performance metrics.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from pathlib import Path
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrendAnalysis:
    """Analysis of demotion and oscillation trends over time."""
    
    period_start: float                      # Start timestamp
    period_end: float                        # End timestamp
    total_demotions: int                     # Total demotions in period
    total_oscillations: int                  # Total oscillations in period
    
    # Trend metrics
    demotion_rate: float                     # Demotions per epoch
    oscillation_rate: float                  # Oscillations per epoch  
    stability_score: float                   # Overall stability (0-1)
    
    # Pattern analysis
    most_demoted_chunks: List[Tuple[str, int]] = field(default_factory=list)  # (chunk_id, count)
    strategy_breakdown: Dict[str, int] = field(default_factory=dict)         # strategy -> count
    risk_distribution: Dict[str, int] = field(default_factory=dict)          # risk_level -> count
    
    # Performance impact
    avg_utility_lost: float = 0.0           # Average utility lost per demotion
    avg_tokens_freed: float = 0.0           # Average tokens freed per demotion
    budget_efficiency: float = 0.0          # Budget utilization efficiency
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get summary metrics for this trend analysis."""
        period_duration = max(1.0, self.period_end - self.period_start)
        
        return {
            'period_duration_hours': period_duration / 3600.0,
            'demotion_rate': self.demotion_rate,
            'oscillation_rate': self.oscillation_rate,
            'stability_score': self.stability_score,
            'total_events': self.total_demotions + self.total_oscillations,
            'avg_utility_lost': self.avg_utility_lost,
            'avg_tokens_freed': self.avg_tokens_freed,
            'budget_efficiency': self.budget_efficiency,
        }


@dataclass
class AnalyticsReport:
    """Comprehensive analytics report for V3 demotion system."""
    
    report_timestamp: float = field(default_factory=time.time)
    analysis_period: Tuple[float, float] = field(default_factory=lambda: (0.0, time.time()))
    
    # Core metrics
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    trend_analysis: Optional[TrendAnalysis] = None
    
    # Detailed breakdowns
    chunk_analysis: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    strategy_analysis: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    epoch_analysis: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Export metadata
    export_paths: Dict[str, str] = field(default_factory=dict)  # type -> path
    
    def add_recommendation(self, text: str, priority: str = 'medium'):
        """Add a recommendation to the report."""
        formatted_rec = f"[{priority.upper()}] {text}"
        self.recommendations.append(formatted_rec)
    
    def add_warning(self, text: str):
        """Add a warning to the report."""
        formatted_warning = f"⚠️  {text}"
        self.warnings.append(formatted_warning)
    
    def get_executive_summary(self) -> Dict[str, Any]:
        """Get executive summary of the analytics report."""
        return {
            'report_timestamp': self.report_timestamp,
            'analysis_period_hours': (
                (self.analysis_period[1] - self.analysis_period[0]) / 3600.0
            ),
            'key_metrics': self.summary_stats,
            'trend_summary': self.trend_analysis.get_summary_metrics() if self.trend_analysis else None,
            'total_recommendations': len(self.recommendations),
            'total_warnings': len(self.warnings),
            'exports_generated': list(self.export_paths.keys()),
        }


class DemotionAnalyzer:
    """
    Comprehensive demotion analytics system for V3 controller.
    
    Analyzes demotion patterns, oscillation trends, and system performance
    to provide insights and recommendations for optimization.
    """
    
    def __init__(self):
        """Initialize demotion analyzer."""
        self._analysis_history: List[AnalyticsReport] = []
        self._chunk_performance_cache: Dict[str, Dict] = {}
        
    def analyze_demotion_data(
        self,
        demotion_decisions: List[Any],  # DemotionDecision objects
        oscillation_events: List[Any],  # OscillationEvent objects
        stability_metrics: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        analysis_period: Optional[Tuple[float, float]] = None
    ) -> AnalyticsReport:
        """
        Perform comprehensive analysis of demotion data.
        
        Args:
            demotion_decisions: List of demotion decisions to analyze
            oscillation_events: List of oscillation events to analyze
            stability_metrics: Stability tracking metrics
            performance_metrics: System performance metrics
            analysis_period: Optional time period for analysis
            
        Returns:
            Comprehensive analytics report
        """
        if analysis_period is None:
            end_time = time.time()
            start_time = end_time - 3600.0  # Last hour by default
            analysis_period = (start_time, end_time)
        
        logger.info(f"Analyzing demotion data for period {analysis_period}")
        
        report = AnalyticsReport(analysis_period=analysis_period)
        
        # Filter data to analysis period
        filtered_demotions = self._filter_by_period(demotion_decisions, analysis_period)
        filtered_oscillations = self._filter_by_period(oscillation_events, analysis_period)
        
        # Generate summary statistics
        report.summary_stats = self._generate_summary_stats(
            filtered_demotions, filtered_oscillations, stability_metrics, performance_metrics
        )
        
        # Perform trend analysis
        report.trend_analysis = self._analyze_trends(
            filtered_demotions, filtered_oscillations, analysis_period
        )
        
        # Detailed analyses
        report.chunk_analysis = self._analyze_chunks(filtered_demotions, filtered_oscillations)
        report.strategy_analysis = self._analyze_strategies(filtered_demotions)
        report.epoch_analysis = self._analyze_epochs(filtered_demotions, filtered_oscillations)
        
        # Generate recommendations and warnings
        self._generate_recommendations(report)
        self._generate_warnings(report)
        
        # Store in history
        self._analysis_history.append(report)
        
        # Keep only recent history to prevent memory growth
        if len(self._analysis_history) > 100:
            self._analysis_history = self._analysis_history[-100:]
        
        logger.info(
            f"Analysis complete: {len(filtered_demotions)} demotions, "
            f"{len(filtered_oscillations)} oscillations analyzed"
        )
        
        return report
    
    def export_analytics_csv(
        self,
        report: AnalyticsReport,
        output_dir: str = "."
    ) -> Dict[str, str]:
        """
        Export analytics data to CSV files.
        
        Args:
            report: Analytics report to export
            output_dir: Output directory for CSV files
            
        Returns:
            Dictionary mapping export type to file path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        export_paths = {}
        
        # Export summary metrics
        summary_path = output_path / "v3_analytics_summary.csv"
        self._export_summary_csv(report, summary_path)
        export_paths['summary'] = str(summary_path)
        
        # Export chunk analysis
        chunk_path = output_path / "v3_chunk_analysis.csv"
        self._export_chunk_analysis_csv(report, chunk_path)
        export_paths['chunks'] = str(chunk_path)
        
        # Export strategy breakdown
        strategy_path = output_path / "v3_strategy_analysis.csv"
        self._export_strategy_analysis_csv(report, strategy_path)
        export_paths['strategies'] = str(strategy_path)
        
        # Export epoch timeline
        epoch_path = output_path / "v3_epoch_timeline.csv"
        self._export_epoch_analysis_csv(report, epoch_path)
        export_paths['epochs'] = str(epoch_path)
        
        # Update report with export paths
        report.export_paths = export_paths
        
        logger.info(f"Exported {len(export_paths)} analytics CSV files to {output_dir}")
        return export_paths
    
    def export_analytics_json(
        self,
        report: AnalyticsReport,
        output_path: str
    ) -> str:
        """
        Export complete analytics report as JSON.
        
        Args:
            report: Analytics report to export
            output_path: Path for JSON output file
            
        Returns:
            Path to exported JSON file
        """
        # Convert report to serializable dictionary
        report_data = {
            'report_timestamp': report.report_timestamp,
            'analysis_period': report.analysis_period,
            'summary_stats': report.summary_stats,
            'trend_analysis': (
                report.trend_analysis.__dict__ if report.trend_analysis else None
            ),
            'chunk_analysis': report.chunk_analysis,
            'strategy_analysis': report.strategy_analysis,
            'epoch_analysis': report.epoch_analysis,
            'recommendations': report.recommendations,
            'warnings': report.warnings,
            'export_paths': report.export_paths,
            'executive_summary': report.get_executive_summary(),
        }
        
        # Write JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Exported complete analytics report to {output_path}")
        return output_path
    
    def get_performance_trends(
        self,
        lookback_reports: int = 10
    ) -> Dict[str, List[float]]:
        """
        Get performance trends across recent analytics reports.
        
        Args:
            lookback_reports: Number of recent reports to analyze
            
        Returns:
            Dictionary with trend data for key metrics
        """
        recent_reports = self._analysis_history[-lookback_reports:]
        
        trends = {
            'stability_scores': [],
            'demotion_rates': [],
            'oscillation_rates': [],
            'budget_efficiencies': [],
            'avg_tokens_freed': [],
        }
        
        for report in recent_reports:
            if report.trend_analysis:
                trends['stability_scores'].append(report.trend_analysis.stability_score)
                trends['demotion_rates'].append(report.trend_analysis.demotion_rate)
                trends['oscillation_rates'].append(report.trend_analysis.oscillation_rate)
                trends['budget_efficiencies'].append(report.trend_analysis.budget_efficiency)
                trends['avg_tokens_freed'].append(report.trend_analysis.avg_tokens_freed)
        
        return trends
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get summary of all analytics performed."""
        return {
            'total_reports': len(self._analysis_history),
            'cached_chunks': len(self._chunk_performance_cache),
            'latest_report_timestamp': (
                self._analysis_history[-1].report_timestamp 
                if self._analysis_history else None
            ),
            'performance_trends': self.get_performance_trends(5),
        }
    
    # Private implementation methods
    
    def _filter_by_period(
        self,
        events: List[Any],
        period: Tuple[float, float]
    ) -> List[Any]:
        """Filter events by time period."""
        start_time, end_time = period
        filtered = []
        
        for event in events:
            event_time = getattr(event, 'timestamp', 0.0)
            if start_time <= event_time <= end_time:
                filtered.append(event)
        
        return filtered
    
    def _generate_summary_stats(
        self,
        demotions: List[Any],
        oscillations: List[Any],
        stability_metrics: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            'total_demotions': len(demotions),
            'total_oscillations': len(oscillations),
            'unique_chunks_affected': len(set(
                getattr(d, 'chunk_id', '') for d in demotions + oscillations
            )),
            'stability_score': stability_metrics.get('stability_score', 0.0),
            'total_tokens_freed': sum(
                getattr(d, 'tokens_freed', 0) for d in demotions
            ),
            'avg_utility_lost': np.mean([
                getattr(d, 'original_utility', 0) - getattr(d, 'recomputed_utility', 0)
                for d in demotions
            ]) if demotions else 0.0,
        }
    
    def _analyze_trends(
        self,
        demotions: List[Any],
        oscillations: List[Any],
        period: Tuple[float, float]
    ) -> TrendAnalysis:
        """Analyze trends in demotion and oscillation data."""
        start_time, end_time = period
        period_duration = max(1.0, end_time - start_time)
        
        # Calculate rates
        demotion_rate = len(demotions) / (period_duration / 3600.0)  # per hour
        oscillation_rate = len(oscillations) / (period_duration / 3600.0)  # per hour
        
        # Calculate stability score
        total_events = len(demotions) + len(oscillations)
        stability_score = max(0.0, 1.0 - (total_events / 100.0))  # Rough heuristic
        
        # Pattern analysis
        chunk_counts = Counter(getattr(d, 'chunk_id', '') for d in demotions)
        most_demoted = chunk_counts.most_common(10)
        
        strategy_counts = Counter(
            getattr(d, 'strategy', {}).get('value', 'unknown') if hasattr(getattr(d, 'strategy', {}), 'get')
            else str(getattr(d, 'strategy', 'unknown'))
            for d in demotions
        )
        
        # Performance metrics
        avg_utility_lost = np.mean([
            getattr(d, 'original_utility', 0) - getattr(d, 'recomputed_utility', 0)
            for d in demotions
        ]) if demotions else 0.0
        
        avg_tokens_freed = np.mean([
            getattr(d, 'tokens_freed', 0) for d in demotions
        ]) if demotions else 0.0
        
        return TrendAnalysis(
            period_start=start_time,
            period_end=end_time,
            total_demotions=len(demotions),
            total_oscillations=len(oscillations),
            demotion_rate=demotion_rate,
            oscillation_rate=oscillation_rate,
            stability_score=stability_score,
            most_demoted_chunks=most_demoted,
            strategy_breakdown=dict(strategy_counts),
            avg_utility_lost=avg_utility_lost,
            avg_tokens_freed=avg_tokens_freed,
        )
    
    def _analyze_chunks(
        self,
        demotions: List[Any],
        oscillations: List[Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze patterns by chunk."""
        chunk_analysis = {}
        
        # Group by chunk ID
        chunk_demotions = defaultdict(list)
        chunk_oscillations = defaultdict(list)
        
        for d in demotions:
            chunk_id = getattr(d, 'chunk_id', '')
            chunk_demotions[chunk_id].append(d)
        
        for o in oscillations:
            chunk_id = getattr(o, 'chunk_id', '')
            chunk_oscillations[chunk_id].append(o)
        
        # Analyze each chunk
        all_chunks = set(chunk_demotions.keys()) | set(chunk_oscillations.keys())
        
        for chunk_id in all_chunks:
            demots = chunk_demotions[chunk_id]
            oscils = chunk_oscillations[chunk_id]
            
            chunk_analysis[chunk_id] = {
                'total_demotions': len(demots),
                'total_oscillations': len(oscils),
                'stability_risk': 'high' if len(oscils) > 2 else 'medium' if len(oscils) > 0 else 'low',
                'avg_tokens_freed': np.mean([getattr(d, 'tokens_freed', 0) for d in demots]) if demots else 0,
                'strategies_used': list(set(str(getattr(d, 'strategy', '')) for d in demots)),
            }
        
        return chunk_analysis
    
    def _analyze_strategies(self, demotions: List[Any]) -> Dict[str, Dict[str, Any]]:
        """Analyze patterns by demotion strategy."""
        strategy_analysis = {}
        
        # Group by strategy
        strategy_demotions = defaultdict(list)
        
        for d in demotions:
            strategy = str(getattr(d, 'strategy', 'unknown'))
            strategy_demotions[strategy].append(d)
        
        # Analyze each strategy
        for strategy, demots in strategy_demotions.items():
            strategy_analysis[strategy] = {
                'count': len(demots),
                'avg_tokens_freed': np.mean([getattr(d, 'tokens_freed', 0) for d in demots]),
                'avg_utility_lost': np.mean([
                    getattr(d, 'original_utility', 0) - getattr(d, 'recomputed_utility', 0)
                    for d in demots
                ]),
                'efficiency_ratio': np.mean([
                    getattr(d, 'tokens_freed', 1) / max(1, 
                        getattr(d, 'original_utility', 0) - getattr(d, 'recomputed_utility', 0)
                    ) for d in demots
                ]),
            }
        
        return strategy_analysis
    
    def _analyze_epochs(
        self,
        demotions: List[Any],
        oscillations: List[Any]
    ) -> Dict[int, Dict[str, Any]]:
        """Analyze patterns by epoch."""
        epoch_analysis = {}
        
        # Group by epoch
        epoch_events = defaultdict(lambda: {'demotions': [], 'oscillations': []})
        
        for d in demotions:
            epoch = getattr(d, 'timestamp', 0)  # Use timestamp as proxy for epoch
            epoch_key = int(epoch // 3600)  # Hour-based epochs
            epoch_events[epoch_key]['demotions'].append(d)
        
        for o in oscillations:
            epoch = getattr(o, 'timestamp', 0)
            epoch_key = int(epoch // 3600)
            epoch_events[epoch_key]['oscillations'].append(o)
        
        # Analyze each epoch
        for epoch_key, events in epoch_events.items():
            demots = events['demotions']
            oscils = events['oscillations']
            
            epoch_analysis[epoch_key] = {
                'demotions_count': len(demots),
                'oscillations_count': len(oscils),
                'total_events': len(demots) + len(oscils),
                'stability_score': max(0.0, 1.0 - (len(demots) + len(oscils)) / 20.0),
                'tokens_freed': sum(getattr(d, 'tokens_freed', 0) for d in demots),
            }
        
        return epoch_analysis
    
    def _generate_recommendations(self, report: AnalyticsReport):
        """Generate recommendations based on analysis."""
        if not report.trend_analysis:
            return
        
        trend = report.trend_analysis
        
        # High demotion rate
        if trend.demotion_rate > 10.0:  # More than 10 per hour
            report.add_recommendation(
                "High demotion rate detected. Consider adjusting utility thresholds "
                "or improving initial selection quality.", 'high'
            )
        
        # High oscillation rate
        if trend.oscillation_rate > 2.0:  # More than 2 per hour
            report.add_recommendation(
                "High oscillation rate detected. Consider increasing ban durations "
                "or implementing more conservative demotion strategies.", 'high'
            )
        
        # Low stability score
        if trend.stability_score < 0.7:
            report.add_recommendation(
                "Low stability score. Review demotion thresholds and consider "
                "implementing stricter oscillation prevention measures.", 'medium'
            )
        
        # Budget efficiency
        if trend.budget_efficiency < 0.8:
            report.add_recommendation(
                "Budget utilization could be improved. Consider optimizing "
                "corrective step algorithms or demotion strategies.", 'medium'
            )
    
    def _generate_warnings(self, report: AnalyticsReport):
        """Generate warnings based on analysis."""
        # High risk chunks
        high_risk_chunks = [
            chunk_id for chunk_id, analysis in report.chunk_analysis.items()
            if analysis['stability_risk'] == 'high'
        ]
        
        if len(high_risk_chunks) > 5:
            report.add_warning(
                f"{len(high_risk_chunks)} chunks have high stability risk. "
                "Consider preemptive banning or strategy adjustments."
            )
        
        # Strategy imbalance
        if report.strategy_analysis:
            strategy_counts = {k: v['count'] for k, v in report.strategy_analysis.items()}
            total_strategies = len(strategy_counts)
            max_strategy_ratio = max(strategy_counts.values()) / sum(strategy_counts.values())
            
            if total_strategies > 1 and max_strategy_ratio > 0.8:
                report.add_warning(
                    "Strategy usage is highly imbalanced. Consider reviewing "
                    "demotion detection algorithms for better strategy diversity."
                )
    
    def _export_summary_csv(self, report: AnalyticsReport, output_path: Path):
        """Export summary metrics to CSV."""
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value', 'Description'])
            
            for key, value in report.summary_stats.items():
                description = self._get_metric_description(key)
                writer.writerow([key, value, description])
    
    def _export_chunk_analysis_csv(self, report: AnalyticsReport, output_path: Path):
        """Export chunk analysis to CSV."""
        if not report.chunk_analysis:
            return
            
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['chunk_id', 'total_demotions', 'total_oscillations', 
                         'stability_risk', 'avg_tokens_freed', 'strategies_used']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for chunk_id, analysis in report.chunk_analysis.items():
                row = analysis.copy()
                row['chunk_id'] = chunk_id
                row['strategies_used'] = ';'.join(row['strategies_used'])
                writer.writerow(row)
    
    def _export_strategy_analysis_csv(self, report: AnalyticsReport, output_path: Path):
        """Export strategy analysis to CSV."""
        if not report.strategy_analysis:
            return
            
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['strategy', 'count', 'avg_tokens_freed', 
                         'avg_utility_lost', 'efficiency_ratio']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for strategy, analysis in report.strategy_analysis.items():
                row = analysis.copy()
                row['strategy'] = strategy
                writer.writerow(row)
    
    def _export_epoch_analysis_csv(self, report: AnalyticsReport, output_path: Path):
        """Export epoch analysis to CSV."""
        if not report.epoch_analysis:
            return
            
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['epoch', 'demotions_count', 'oscillations_count', 
                         'total_events', 'stability_score', 'tokens_freed']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for epoch, analysis in sorted(report.epoch_analysis.items()):
                row = analysis.copy()
                row['epoch'] = epoch
                writer.writerow(row)
    
    def _get_metric_description(self, metric: str) -> str:
        """Get description for a metric."""
        descriptions = {
            'total_demotions': 'Total number of chunk demotions',
            'total_oscillations': 'Total number of detected oscillations',
            'unique_chunks_affected': 'Number of unique chunks involved in demotions/oscillations',
            'stability_score': 'Overall system stability score (0-1)',
            'total_tokens_freed': 'Total tokens freed by demotions',
            'avg_utility_lost': 'Average utility lost per demotion',
        }
        return descriptions.get(metric, 'No description available')
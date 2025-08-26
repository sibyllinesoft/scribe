"""
V3 Demotion Visualization System

Provides visualization tools for demotion patterns, oscillation trends,
and system performance metrics (placeholder implementation).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class OscillationChart:
    """Chart data for oscillation visualization."""
    
    chart_type: str = "line"                 # Chart type
    title: str = "Oscillation Trends"       # Chart title
    x_axis_label: str = "Time"              # X-axis label
    y_axis_label: str = "Oscillations"     # Y-axis label
    
    # Data points
    timestamps: List[float] = field(default_factory=list)
    oscillation_counts: List[int] = field(default_factory=list)
    chunk_labels: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chart to dictionary for JSON serialization."""
        return {
            'chart_type': self.chart_type,
            'title': self.title,
            'x_axis_label': self.x_axis_label,
            'y_axis_label': self.y_axis_label,
            'data': {
                'timestamps': self.timestamps,
                'oscillation_counts': self.oscillation_counts,
                'chunk_labels': self.chunk_labels,
            }
        }


@dataclass  
class PerformanceMetrics:
    """Performance metrics visualization data."""
    
    metric_name: str                         # Name of metric
    current_value: float                     # Current value
    target_value: Optional[float] = None     # Target value if applicable
    trend_data: List[float] = field(default_factory=list)  # Historical trend data
    trend_timestamps: List[float] = field(default_factory=list)  # Trend timestamps
    
    # Metadata
    unit: str = ""                          # Unit of measurement
    description: str = ""                   # Metric description
    status: str = "normal"                  # normal, warning, critical
    
    def get_trend_direction(self) -> str:
        """Get trend direction based on recent data."""
        if len(self.trend_data) < 2:
            return "stable"
        
        recent_avg = sum(self.trend_data[-3:]) / len(self.trend_data[-3:])
        earlier_avg = sum(self.trend_data[-6:-3]) / max(1, len(self.trend_data[-6:-3]))
        
        if recent_avg > earlier_avg * 1.1:
            return "increasing"
        elif recent_avg < earlier_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'target_value': self.target_value,
            'trend_data': self.trend_data,
            'trend_timestamps': self.trend_timestamps,
            'unit': self.unit,
            'description': self.description,
            'status': self.status,
            'trend_direction': self.get_trend_direction(),
        }


class DemotionVisualizer:
    """
    Visualization system for V3 demotion analytics.
    
    Provides methods to create charts and visualizations for demotion patterns,
    oscillation trends, and performance metrics. This is a placeholder implementation
    that generates visualization data structures that could be used with plotting libraries.
    """
    
    def __init__(self):
        """Initialize demotion visualizer."""
        self._generated_charts: List[Dict[str, Any]] = []
        
    def create_oscillation_timeline(
        self,
        oscillation_events: List[Any],
        chunk_filter: Optional[List[str]] = None,
        time_window: Optional[Tuple[float, float]] = None
    ) -> OscillationChart:
        """
        Create oscillation timeline chart.
        
        Args:
            oscillation_events: List of oscillation events to visualize
            chunk_filter: Optional list of chunk IDs to include
            time_window: Optional time window to focus on
            
        Returns:
            Oscillation chart data structure
        """
        # Filter events if needed
        filtered_events = oscillation_events
        
        if time_window:
            start_time, end_time = time_window
            filtered_events = [
                e for e in filtered_events
                if start_time <= getattr(e, 'timestamp', 0) <= end_time
            ]
        
        if chunk_filter:
            filtered_events = [
                e for e in filtered_events
                if getattr(e, 'chunk_id', '') in chunk_filter
            ]
        
        # Extract data for chart
        timestamps = [getattr(e, 'timestamp', 0) for e in filtered_events]
        chunk_labels = [getattr(e, 'chunk_id', 'unknown') for e in filtered_events]
        
        # Create cumulative oscillation counts
        oscillation_counts = list(range(1, len(filtered_events) + 1))
        
        chart = OscillationChart(
            timestamps=timestamps,
            oscillation_counts=oscillation_counts,
            chunk_labels=chunk_labels
        )
        
        self._generated_charts.append(chart.to_dict())
        logger.info(f"Created oscillation timeline with {len(filtered_events)} events")
        
        return chart
    
    def create_demotion_strategy_breakdown(
        self,
        demotion_decisions: List[Any],
        strategy_analysis: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create demotion strategy breakdown visualization.
        
        Args:
            demotion_decisions: List of demotion decisions
            strategy_analysis: Strategy analysis data
            
        Returns:
            Strategy breakdown chart data
        """
        # Prepare pie chart data
        strategy_counts = {}
        strategy_efficiencies = {}
        
        for strategy, analysis in strategy_analysis.items():
            strategy_counts[strategy] = analysis.get('count', 0)
            strategy_efficiencies[strategy] = analysis.get('efficiency_ratio', 0.0)
        
        chart_data = {
            'chart_type': 'pie',
            'title': 'Demotion Strategy Breakdown',
            'data': {
                'labels': list(strategy_counts.keys()),
                'counts': list(strategy_counts.values()),
                'efficiencies': list(strategy_efficiencies.values()),
            },
            'metadata': {
                'total_demotions': sum(strategy_counts.values()),
                'most_used_strategy': max(strategy_counts.keys(), key=lambda k: strategy_counts[k]),
                'most_efficient_strategy': max(strategy_efficiencies.keys(), key=lambda k: strategy_efficiencies[k]),
            }
        }
        
        self._generated_charts.append(chart_data)
        logger.info(f"Created strategy breakdown chart with {len(strategy_counts)} strategies")
        
        return chart_data
    
    def create_performance_dashboard(
        self,
        performance_trends: Dict[str, List[float]],
        current_metrics: Dict[str, Any]
    ) -> List[PerformanceMetrics]:
        """
        Create performance metrics dashboard.
        
        Args:
            performance_trends: Historical trend data for metrics
            current_metrics: Current metric values
            
        Returns:
            List of performance metric visualizations
        """
        dashboard_metrics = []
        
        # Define target values and descriptions for key metrics
        metric_configs = {
            'stability_score': {
                'target': 0.9,
                'unit': 'score',
                'description': 'Overall system stability (higher is better)',
            },
            'demotion_rate': {
                'target': 5.0,
                'unit': 'per hour',
                'description': 'Rate of chunk demotions (lower is better)',
            },
            'oscillation_rate': {
                'target': 1.0,
                'unit': 'per hour', 
                'description': 'Rate of oscillations detected (lower is better)',
            },
            'budget_efficiency': {
                'target': 0.95,
                'unit': 'ratio',
                'description': 'Budget utilization efficiency (higher is better)',
            },
            'avg_tokens_freed': {
                'target': None,
                'unit': 'tokens',
                'description': 'Average tokens freed per demotion',
            },
        }
        
        # Create metrics for each tracked value
        for metric_name, trend_data in performance_trends.items():
            config = metric_configs.get(metric_name, {})
            current_value = current_metrics.get(metric_name, 0.0)
            
            # Determine status
            target = config.get('target')
            status = 'normal'
            if target:
                if metric_name in ['stability_score', 'budget_efficiency']:
                    # Higher is better
                    if current_value < target * 0.8:
                        status = 'critical'
                    elif current_value < target * 0.9:
                        status = 'warning'
                else:
                    # Lower is better
                    if current_value > target * 2.0:
                        status = 'critical'
                    elif current_value > target * 1.5:
                        status = 'warning'
            
            # Generate timestamps (placeholder - in real implementation would use actual timestamps)
            timestamps = [time.time() - (len(trend_data) - i) * 3600 for i in range(len(trend_data))]
            
            metric = PerformanceMetrics(
                metric_name=metric_name,
                current_value=current_value,
                target_value=target,
                trend_data=trend_data,
                trend_timestamps=timestamps,
                unit=config.get('unit', ''),
                description=config.get('description', ''),
                status=status
            )
            
            dashboard_metrics.append(metric)
        
        # Store dashboard data
        dashboard_data = {
            'chart_type': 'dashboard',
            'title': 'V3 Performance Dashboard',
            'metrics': [m.to_dict() for m in dashboard_metrics],
            'generated_at': time.time(),
        }
        
        self._generated_charts.append(dashboard_data)
        logger.info(f"Created performance dashboard with {len(dashboard_metrics)} metrics")
        
        return dashboard_metrics
    
    def create_chunk_risk_heatmap(
        self,
        chunk_analysis: Dict[str, Dict[str, Any]],
        max_chunks: int = 50
    ) -> Dict[str, Any]:
        """
        Create chunk risk level heatmap.
        
        Args:
            chunk_analysis: Per-chunk analysis data
            max_chunks: Maximum number of chunks to include
            
        Returns:
            Heatmap chart data
        """
        # Sort chunks by risk level and activity
        sorted_chunks = sorted(
            chunk_analysis.items(),
            key=lambda x: (
                {'high': 3, 'medium': 2, 'low': 1}.get(x[1].get('stability_risk', 'low'), 0),
                x[1].get('total_demotions', 0) + x[1].get('total_oscillations', 0)
            ),
            reverse=True
        )
        
        # Take top chunks
        top_chunks = sorted_chunks[:max_chunks]
        
        # Prepare heatmap data
        chunk_ids = [chunk_id for chunk_id, _ in top_chunks]
        risk_levels = [analysis.get('stability_risk', 'low') for _, analysis in top_chunks]
        activity_scores = [
            analysis.get('total_demotions', 0) + analysis.get('total_oscillations', 0)
            for _, analysis in top_chunks
        ]
        
        # Convert risk levels to numeric values
        risk_numeric = [
            {'high': 3, 'medium': 2, 'low': 1}.get(risk, 1) for risk in risk_levels
        ]
        
        heatmap_data = {
            'chart_type': 'heatmap',
            'title': 'Chunk Risk Level Heatmap',
            'x_axis_label': 'Chunks',
            'y_axis_label': 'Risk Level',
            'data': {
                'chunk_ids': chunk_ids,
                'risk_levels': risk_levels,
                'risk_numeric': risk_numeric,
                'activity_scores': activity_scores,
            },
            'color_scale': {
                'low': '#90EE90',      # Light green
                'medium': '#FFD700',   # Gold  
                'high': '#FF6B6B',     # Light red
            }
        }
        
        self._generated_charts.append(heatmap_data)
        logger.info(f"Created chunk risk heatmap with {len(chunk_ids)} chunks")
        
        return heatmap_data
    
    def export_visualization_data(self, output_path: str) -> str:
        """
        Export all generated visualization data to JSON.
        
        Args:
            output_path: Path to export JSON file
            
        Returns:
            Path to exported file
        """
        visualization_package = {
            'generated_at': time.time(),
            'total_charts': len(self._generated_charts),
            'charts': self._generated_charts,
            'metadata': {
                'generator': 'PackRepo V3 DemotionVisualizer',
                'version': '1.0',
                'description': 'Visualization data for V3 demotion analytics',
            }
        }
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(visualization_package, f, indent=2, default=str)
        
        logger.info(f"Exported {len(self._generated_charts)} visualizations to {output_path}")
        return output_path
    
    def get_chart_summary(self) -> Dict[str, Any]:
        """Get summary of generated charts."""
        chart_types = {}
        for chart in self._generated_charts:
            chart_type = chart.get('chart_type', 'unknown')
            chart_types[chart_type] = chart_types.get(chart_type, 0) + 1
        
        return {
            'total_charts': len(self._generated_charts),
            'chart_types': chart_types,
            'latest_generation': max(
                (chart.get('generated_at', 0) for chart in self._generated_charts),
                default=0
            )
        }
    
    def clear_charts(self):
        """Clear all generated chart data."""
        self._generated_charts.clear()
        logger.info("Cleared all generated visualization data")
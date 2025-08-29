"""
Visualization tools for SpectraVision analysis results
Creates charts, graphs, and dashboards for performance analysis
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    plt = sns = go = px = make_subplots = None

logger = logging.getLogger(__name__)

class SpectraVisionVisualizer:
    """Visualization generator for SpectraVision analysis results"""
    
    def __init__(self, results_dir: str, style: str = "seaborn"):
        """
        Initialize visualizer
        
        Args:
            results_dir: Directory containing analysis results
            style: Matplotlib style to use
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("  Visualization libraries not available. Install matplotlib, seaborn, plotly")
            self.enabled = False
            return
        
        self.results_dir = Path(results_dir)
        self.enabled = True
        
        # Setup matplotlib style
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Load data
        self.results_data = self._load_results()
        self.monitoring_data = self._load_monitoring_data()
        self.analysis_data = self._load_analysis_data()
        
        # Output directory for visualizations
        self.viz_dir = self.results_dir / "analysis" / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(" SpectraVision Visualizer initialized")
    
    def _load_results(self) -> Optional[Dict[str, Any]]:
        """Load evaluation results"""
        results_file = self.results_dir / "results" / "final_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        return None
    
    def _load_monitoring_data(self) -> Optional[Dict[str, Any]]:
        """Load monitoring data"""
        monitoring_file = self.results_dir / "reports" / "performance_monitoring.json"
        if monitoring_file.exists():
            with open(monitoring_file, 'r') as f:
                return json.load(f)
        return None
    
    def _load_analysis_data(self) -> Optional[Dict[str, Any]]:
        """Load analysis data"""
        analysis_files = list((self.results_dir / "analysis").glob("detailed_analysis_*.json"))
        if analysis_files:
            # Use most recent analysis file
            latest_file = max(analysis_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, 'r') as f:
                return json.load(f)
        return None
    
    def create_overview_dashboard(self) -> str:
        """
        Create comprehensive overview dashboard
        
        Returns:
            Path to generated HTML dashboard
        """
        if not self.enabled or not self.results_data:
            return ""
        
        logger.info(" Creating overview dashboard...")
        
        evaluations = self.results_data.get('evaluations', [])
        df = pd.DataFrame(evaluations)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Success Rate by Model",
                "Execution Time Distribution", 
                "Memory Usage by Model-Benchmark",
                "Performance Heatmap"
            ],
            specs=[
                [{"type": "bar"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "heatmap"}]
            ]
        )
        
        # 1. Success rate by model
        success_by_model = df.groupby('model_name')['success'].agg(['sum', 'count']).reset_index()
        success_by_model['success_rate'] = success_by_model['sum'] / success_by_model['count'] * 100
        
        fig.add_trace(
            go.Bar(
                x=success_by_model['model_name'],
                y=success_by_model['success_rate'],
                name="Success Rate",
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # 2. Execution time distribution
        successful_evals = df[df['success'] == True]
        fig.add_trace(
            go.Histogram(
                x=successful_evals['execution_time'],
                nbinsx=20,
                name="Execution Time",
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
        
        # 3. Memory usage scatter plot
        fig.add_trace(
            go.Scatter(
                x=successful_evals['execution_time'],
                y=successful_evals['peak_memory'],
                mode='markers',
                text=successful_evals['model_name'] + ' - ' + successful_evals['benchmark_name'],
                name="Memory vs Time",
                marker=dict(size=8, opacity=0.7)
            ),
            row=2, col=1
        )
        
        # 4. Performance heatmap
        if len(df) > 0:
            pivot_data = df.pivot_table(
                values='execution_time', 
                index='model_name', 
                columns='benchmark_name', 
                aggfunc='mean',
                fill_value=0
            )
            
            fig.add_trace(
                go.Heatmap(
                    z=pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    colorscale='Viridis',
                    name="Avg Execution Time"
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="SpectraVision Performance Dashboard",
            title_x=0.5,
            showlegend=False
        )
        
        # Save dashboard
        dashboard_file = self.viz_dir / "overview_dashboard.html"
        fig.write_html(str(dashboard_file))
        
        logger.info(f" Dashboard saved to: {dashboard_file}")
        return str(dashboard_file)
    
    def create_model_comparison_chart(self) -> str:
        """Create detailed model comparison visualization"""
        if not self.enabled or not self.results_data:
            return ""
        
        logger.info(" Creating model comparison chart...")
        
        evaluations = self.results_data.get('evaluations', [])
        df = pd.DataFrame(evaluations)
        
        # Calculate model statistics
        model_stats = []
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            
            stats = {
                'model': model,
                'success_rate': (model_data['success'].sum() / len(model_data)) * 100,
                'avg_time': model_data[model_data['success']]['execution_time'].mean(),
                'avg_memory': model_data[model_data['success']]['peak_memory'].mean(),
                'total_evaluations': len(model_data)
            }
            model_stats.append(stats)
        
        stats_df = pd.DataFrame(model_stats)
        
        # Create multi-metric comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Success rate
        bars1 = axes[0, 0].bar(stats_df['model'], stats_df['success_rate'], color='skyblue')
        axes[0, 0].set_title('Success Rate (%)')
        axes[0, 0].set_ylim(0, 100)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%', ha='center', va='bottom')
        
        # Average execution time
        bars2 = axes[0, 1].bar(stats_df['model'], stats_df['avg_time'], color='lightcoral')
        axes[0, 1].set_title('Average Execution Time (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Average memory usage
        bars3 = axes[1, 0].bar(stats_df['model'], stats_df['avg_memory'], color='lightgreen')
        axes[1, 0].set_title('Average Peak Memory (GB)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Efficiency scatter plot (time vs memory, size = success rate)
        scatter = axes[1, 1].scatter(
            stats_df['avg_time'], 
            stats_df['avg_memory'],
            s=stats_df['success_rate'] * 5,  # Size proportional to success rate
            alpha=0.7,
            c=range(len(stats_df)),
            cmap='viridis'
        )
        axes[1, 1].set_xlabel('Average Execution Time (seconds)')
        axes[1, 1].set_ylabel('Average Peak Memory (GB)')
        axes[1, 1].set_title('Efficiency Plot (size = success rate)')
        
        # Add model names as labels
        for i, model in enumerate(stats_df['model']):
            axes[1, 1].annotate(model, (stats_df.iloc[i]['avg_time'], stats_df.iloc[i]['avg_memory']),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        # Save chart
        chart_file = self.viz_dir / "model_comparison.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f" Model comparison chart saved to: {chart_file}")
        return str(chart_file)
    
    def create_performance_timeline(self) -> str:
        """Create timeline visualization of evaluation performance"""
        if not self.enabled or not self.results_data:
            return ""
        
        logger.info(" Creating performance timeline...")
        
        evaluations = self.results_data.get('evaluations', [])
        df = pd.DataFrame(evaluations)
        
        # Convert timestamps to datetime
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        
        # Sort by start time
        df = df.sort_values('start_time').reset_index(drop=True)
        df['eval_order'] = range(len(df))
        
        # Create timeline plot
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        fig.suptitle('Evaluation Performance Timeline', fontsize=16, fontweight='bold')
        
        # 1. Execution time over evaluations
        colors = ['green' if success else 'red' for success in df['success']]
        axes[0].scatter(df['eval_order'], df['execution_time'], c=colors, alpha=0.7)
        axes[0].set_ylabel('Execution Time (seconds)')
        axes[0].set_title('Execution Time per Evaluation (Green=Success, Red=Failure)')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Memory usage over evaluations
        successful_df = df[df['success'] == True]
        axes[1].scatter(successful_df['eval_order'], successful_df['peak_memory'], 
                       c='blue', alpha=0.7)
        axes[1].set_ylabel('Peak Memory (GB)')
        axes[1].set_title('Peak Memory Usage per Evaluation')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Success rate rolling window
        window_size = max(5, len(df) // 10)  # Adaptive window size
        rolling_success = df['success'].rolling(window=window_size, min_periods=1).mean() * 100
        axes[2].plot(df['eval_order'], rolling_success, color='purple', linewidth=2)
        axes[2].fill_between(df['eval_order'], rolling_success, alpha=0.3, color='purple')
        axes[2].set_ylabel('Rolling Success Rate (%)')
        axes[2].set_xlabel('Evaluation Order')
        axes[2].set_title(f'Rolling Success Rate (window size: {window_size})')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 100)
        
        plt.tight_layout()
        
        # Save timeline
        timeline_file = self.viz_dir / "performance_timeline.png"
        plt.savefig(timeline_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f" Performance timeline saved to: {timeline_file}")
        return str(timeline_file)
    
    def create_resource_monitoring_charts(self) -> str:
        """Create resource monitoring visualizations"""
        if not self.enabled or not self.monitoring_data:
            logger.warning("No monitoring data available for visualization")
            return ""
        
        logger.info(" Creating resource monitoring charts...")
        
        eval_metrics = self.monitoring_data.get('evaluation_metrics', [])
        if not eval_metrics:
            logger.warning("No evaluation metrics in monitoring data")
            return ""
        
        # Create monitoring dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "GPU Memory Usage Timeline",
                "GPU Temperature Timeline",
                "GPU Utilization Timeline", 
                "Resource Usage Distribution"
            ]
        )
        
        # Process monitoring data
        for i, eval_metric in enumerate(eval_metrics):
            snapshots = eval_metric.get('snapshots', [])
            if not snapshots:
                continue
            
            times = [s['timestamp'] for s in snapshots]
            memories = [s['gpu_memory_used'] for s in snapshots]
            temperatures = [s['gpu_temperature'] for s in snapshots]
            utilizations = [s['gpu_utilization'] for s in snapshots]
            
            model_name = eval_metric.get('model_name', f'Eval {i}')
            
            # GPU Memory
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=memories,
                    mode='lines',
                    name=f'{model_name} Memory',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
            
            # GPU Temperature
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=temperatures,
                    mode='lines',
                    name=f'{model_name} Temp',
                    line=dict(width=2)
                ),
                row=1, col=2
            )
            
            # GPU Utilization
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=utilizations,
                    mode='lines',
                    name=f'{model_name} Util',
                    line=dict(width=2)
                ),
                row=2, col=1
            )
        
        # Resource distribution (box plots)
        all_memories = []
        all_temps = []
        all_utils = []
        model_labels = []
        
        for eval_metric in eval_metrics:
            snapshots = eval_metric.get('snapshots', [])
            if snapshots:
                model_name = eval_metric.get('model_name', 'Unknown')
                for snapshot in snapshots:
                    all_memories.append(snapshot['gpu_memory_used'])
                    all_temps.append(snapshot['gpu_temperature'])
                    all_utils.append(snapshot['gpu_utilization'])
                    model_labels.append(model_name)
        
        if all_memories:
            fig.add_trace(
                go.Box(
                    y=all_memories,
                    x=model_labels,
                    name="Memory Distribution",
                    boxpoints='outliers'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Resource Monitoring Dashboard",
            showlegend=True
        )
        
        # Save monitoring charts
        monitoring_file = self.viz_dir / "resource_monitoring.html"
        fig.write_html(str(monitoring_file))
        
        logger.info(f" Resource monitoring charts saved to: {monitoring_file}")
        return str(monitoring_file)
    
    def generate_all_visualizations(self) -> Dict[str, str]:
        """
        Generate all available visualizations
        
        Returns:
            Dictionary mapping visualization names to file paths
        """
        if not self.enabled:
            logger.warning("Visualization not enabled - missing required libraries")
            return {}
        
        logger.info(" Generating all visualizations...")
        
        visualizations = {}
        
        try:
            # Overview dashboard
            viz_file = self.create_overview_dashboard()
            if viz_file:
                visualizations['overview_dashboard'] = viz_file
            
            # Model comparison
            viz_file = self.create_model_comparison_chart()
            if viz_file:
                visualizations['model_comparison'] = viz_file
            
            # Performance timeline
            viz_file = self.create_performance_timeline()
            if viz_file:
                visualizations['performance_timeline'] = viz_file
            
            # Resource monitoring (if available)
            viz_file = self.create_resource_monitoring_charts()
            if viz_file:
                visualizations['resource_monitoring'] = viz_file
            
            # Create index file
            index_file = self._create_visualization_index(visualizations)
            visualizations['index'] = index_file
            
            logger.info(f" Generated {len(visualizations)} visualizations")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        return visualizations
    
    def _create_visualization_index(self, visualizations: Dict[str, str]) -> str:
        """Create HTML index file for all visualizations"""
        index_file = self.viz_dir / "index.html"
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SpectraVision Analysis Visualizations</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                .viz-link { display: block; margin: 10px 0; padding: 10px; 
                           background-color: #f0f0f0; text-decoration: none; 
                           color: #333; border-radius: 5px; }
                .viz-link:hover { background-color: #e0e0e0; }
            </style>
        </head>
        <body>
            <h1> SpectraVision Analysis Visualizations</h1>
            <p>Generated at: {timestamp}</p>
            
            <h2>Available Visualizations:</h2>
        """.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        for name, path in visualizations.items():
            if name != 'index':
                display_name = name.replace('_', ' ').title()
                file_name = Path(path).name
                html_content += f'<a href="{file_name}" class="viz-link"> {display_name}</a>\n'
        
        html_content += """
            </body>
        </html>
        """
        
        with open(index_file, 'w') as f:
            f.write(html_content)
        
        return str(index_file)
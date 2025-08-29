"""
Analysis module for SpectraVision
Comprehensive performance analysis and visualization tools
"""

from .analyzer import PerformanceAnalyzer, PerformanceInsights, BottleneckAnalysis
from .visualizer import SpectraVisionVisualizer

__all__ = [
    'PerformanceAnalyzer',
    'PerformanceInsights', 
    'BottleneckAnalysis',
    'SpectraVisionVisualizer'
]

# Analysis module metadata
__version__ = "0.1.0"
__description__ = "Performance analysis and visualization tools for SpectraVision"

# Quick analysis function for convenience
def analyze_results(results_dir: str, generate_visualizations: bool = True) -> tuple:
    """
    Quick analysis function - analyzes results and optionally generates visualizations
    
    Args:
        results_dir: Directory containing SpectraVision results
        generate_visualizations: Whether to generate visualization files
        
    Returns:
        Tuple of (report_file_path, visualizations_dict)
    """
    # Perform analysis
    analyzer = PerformanceAnalyzer(results_dir)
    insights = analyzer.analyze_performance()
    report_file, json_file = analyzer.save_analysis_results()
    
    visualizations = {}
    if generate_visualizations:
        # Generate visualizations
        visualizer = SpectraVisionVisualizer(results_dir)
        visualizations = visualizer.generate_all_visualizations()
    
    return report_file, visualizations